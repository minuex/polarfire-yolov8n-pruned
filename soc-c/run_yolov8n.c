#define _POSIX_C_SOURCE 200809L

#include "postprocess_yolov8n.h"
#include "vbx_cnn_api.h"

#include <dirent.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

/* VectorBlox SDK example/postprocess/image.c (-ljpeg). */
extern int read_JPEG_file(const char *filename, int *width, int *height,
                          unsigned char **image, const int grayscale);
extern void resize_image(uint8_t *image_in, int in_w, int in_h,
                         uint8_t *image_out, int out_w, int out_h);

#ifndef USE_INTERRUPTS
#define USE_INTERRUPTS 0
#endif

#define EXPECTED_INPUTS 1U
#define MAX_STANDALONE_DETECTIONS 100

typedef struct {
    vbx_cnn_t *vbx;
    model_t *model;
    vbx_cnn_io_ptr_t io[MAX_IO_BUFFERS];
    yolov8n_output_t outputs[YOLOV8N_NUM_OUTPUTS];
    unsigned input_count;
    unsigned output_count;
    int input_width;
    int input_height;
    int input_is_hcw;
    int output_channels;
    int num_classes;
    float confidence_threshold;
    float nms_iou_threshold;
} yolo_runtime_t;

static yolo_runtime_t g_yolo;

#ifndef YOLOV8N_LIBRARY_ONLY
static double elapsed_ms(const struct timespec *start,
                         const struct timespec *end)
{
    return (double)(end->tv_sec - start->tv_sec) * 1000.0 +
           (double)(end->tv_nsec - start->tv_nsec) / 1000000.0;
}
#endif

/* Overlay/DMA device omission otherwise tends to fail inside the SDK. */
static int check_board_devices(void)
{
    DIR *directory;
    struct dirent *entry;
    int found_uio = 0;

    if (access("/dev/udmabuf-vbx64", R_OK | W_OK) != 0) {
        fprintf(stderr,
                "Missing /dev/udmabuf-vbx64. Apply the VBX overlay and "
                "vbx_dma64.dtbo first.\n");
        return -1;
    }
    directory = opendir("/sys/class/uio");
    if (directory == NULL) {
        fprintf(stderr, "Missing /sys/class/uio. Apply the VBX overlay first.\n");
        return -1;
    }
    while ((entry = readdir(directory)) != NULL) {
        if (strncmp(entry->d_name, "uio", 3) == 0) {
            found_uio = 1;
            break;
        }
    }
    closedir(directory);
    if (!found_uio) {
        fprintf(stderr, "No UIO accelerator device. Apply the VBX overlay first.\n");
        return -1;
    }
    return 0;
}

static model_t *load_model_to_dma(vbx_cnn_t *vbx, const char *path)
{
    FILE *file;
    model_t *host_model = NULL;
    model_t *dma_model = NULL;
    long file_size;
    int data_bytes;
    int allocate_bytes;

    file = fopen(path, "rb");
    if (file == NULL) {
        fprintf(stderr, "Unable to open VNNX model: %s\n", path);
        return NULL;
    }
    if (fseek(file, 0, SEEK_END) != 0 ||
        (file_size = ftell(file)) <= 0 || fseek(file, 0, SEEK_SET) != 0) {
        fprintf(stderr, "Unable to determine VNNX file size: %s\n", path);
        fclose(file);
        return NULL;
    }
    host_model = (model_t *)malloc((size_t)file_size);
    if (host_model == NULL ||
        fread(host_model, 1, (size_t)file_size, file) != (size_t)file_size) {
        fprintf(stderr, "Unable to read complete VNNX model: %s\n", path);
        free(host_model);
        fclose(file);
        return NULL;
    }
    fclose(file);

    data_bytes = model_get_data_bytes(host_model);
    allocate_bytes = model_get_allocate_bytes(host_model);
    printf("DATA_BYTES     = %d (%.2f MiB)\n", data_bytes,
           (double)data_bytes / (1024.0 * 1024.0));
    printf("ALLOCATE_BYTES = %d (%.2f MiB)\n", allocate_bytes,
           (double)allocate_bytes / (1024.0 * 1024.0));
    printf("MODEL_SANITY   = %d\n", model_check_sanity(host_model));
    if (data_bytes != file_size || allocate_bytes < data_bytes ||
        model_check_sanity(host_model) != 0) {
        fprintf(stderr, "Invalid or incompatible VNNX model\n");
        free(host_model);
        return NULL;
    }

    dma_model = (model_t *)vbx_allocate_dma_buffer(
        vbx, (size_t)allocate_bytes, 0);
    if (dma_model == NULL) {
        fprintf(stderr, "Unable to allocate model in the DMA64 pool\n");
        free(host_model);
        return NULL;
    }
    memcpy(dma_model, host_model, (size_t)data_bytes);
    free(host_model);
    return dma_model;
}

static int infer_output_channels(model_t *model, unsigned output_count)
{
    int *first_shape = model_get_output_shape(model, 0);
    int first_dims = model_get_output_dims(model, 0);
    int candidate_index;

    if (first_shape == NULL || (first_dims != 3 && first_dims != 4)) return -1;
    for (candidate_index = 0; candidate_index < first_dims; ++candidate_index) {
        int candidate = first_shape[candidate_index];
        int common = candidate > YOLOV8N_DFL_CHANNELS;
        unsigned output_index;

        for (output_index = 1;
             common && output_index < output_count; ++output_index) {
            int *shape = model_get_output_shape(model, output_index);
            int dims = model_get_output_dims(model, output_index);
            int found = 0;
            int d;
            if (shape == NULL || (dims != 3 && dims != 4)) return -1;
            for (d = 0; d < dims; ++d) {
                if (shape[d] == candidate) found = 1;
            }
            common = found;
        }
        if (common) return candidate;
    }
    return -1;
}

static int get_input_geometry(model_t *model, int *width, int *height,
                              int *is_hcw)
{
    int *shape = model_get_input_shape(model, 0);
    int dims = model_get_input_dims(model, 0);
    int d;

    if (shape == NULL || (dims != 3 && dims != 4)) return -1;
    printf("input raw shape: [");
    for (d = 0; d < dims; ++d) printf("%s%d", d ? "," : "", shape[d]);
    printf("]\n");
    if (dims == 4 && shape[0] != 1) return -1;

    *is_hcw = 0;
    if (shape[dims - 3] == 3) {
        *height = shape[dims - 2];
        *width = shape[dims - 1];
    } else if (shape[dims - 2] == 3) {
        *is_hcw = 1;
        *height = shape[dims - 3];
        *width = shape[dims - 1];
    } else {
        return -1;
    }
    return 0;
}

static uint8_t *preprocess_rgb_planar(const uint8_t *rgb, int width,
                                      int height, int output_width,
                                      int output_height)
{
    uint8_t *planar;
    uint8_t *resized;
    size_t input_plane;
    size_t output_plane;
    int x;
    int y;
    int channel;

    if (rgb == NULL || width <= 0 || height <= 0 ||
        output_width <= 0 || output_height <= 0) return NULL;
    input_plane = (size_t)width * (size_t)height;
    output_plane = (size_t)output_width * (size_t)output_height;
    planar = (uint8_t *)malloc(input_plane * 3U);
    resized = (uint8_t *)malloc(output_plane * 3U);
    if (planar == NULL || resized == NULL) {
        free(planar);
        free(resized);
        return NULL;
    }

    for (y = 0; y < height; ++y) {
        for (x = 0; x < width; ++x) {
            size_t pixel = (size_t)y * (size_t)width + (size_t)x;
            for (channel = 0; channel < 3; ++channel) {
                planar[(size_t)channel * input_plane + pixel] =
                    rgb[pixel * 3U + (size_t)channel];
            }
        }
    }
    for (channel = 0; channel < 3; ++channel) {
        resize_image(planar + (size_t)channel * input_plane,
                     width, height,
                     resized + (size_t)channel * output_plane,
                     output_width, output_height);
    }
    free(planar);
    return resized;
}

static uint8_t *repack_chw_to_hcw(uint8_t *chw, int width, int height)
{
    uint8_t *hcw;
    int y;
    int channel;

    hcw = (uint8_t *)malloc((size_t)width * (size_t)height * 3U);
    if (hcw == NULL) {
        free(chw);
        return NULL;
    }
    for (y = 0; y < height; ++y) {
        for (channel = 0; channel < 3; ++channel) {
            memcpy(hcw + (((size_t)y * 3U + (size_t)channel) * (size_t)width),
                   chw + (((size_t)channel * (size_t)height + (size_t)y) *
                          (size_t)width),
                   (size_t)width);
        }
    }
    free(chw);
    return hcw;
}

static int configure_output(model_t *model, int index, const int8_t *data,
                            int expected_channels, yolov8n_output_t *output)
{
    int *shape = model_get_output_shape(model, index);
    int dims = model_get_output_dims(model, index);
    const char *layout_name;
    int d;

    if (shape == NULL || output == NULL || (dims != 3 && dims != 4)) return -1;
    printf("output[%d] raw shape: [", index);
    for (d = 0; d < dims; ++d) printf("%s%d", d ? "," : "", shape[d]);
    printf("]\n");

    if (dims == 4 && shape[3] == expected_channels) {
        output->layout = YOLOV8N_LAYOUT_NHWC;
        output->height = shape[1]; output->width = shape[2];
        output->channels = shape[3];
    } else if (dims == 4 && shape[1] == expected_channels) {
        output->layout = YOLOV8N_LAYOUT_NCHW;
        output->channels = shape[1]; output->height = shape[2];
        output->width = shape[3];
    } else if (dims == 3 && shape[2] == expected_channels) {
        output->layout = YOLOV8N_LAYOUT_NHWC;
        output->height = shape[0]; output->width = shape[1];
        output->channels = shape[2];
    } else if (dims == 3 && shape[0] == expected_channels) {
        output->layout = YOLOV8N_LAYOUT_NCHW;
        output->channels = shape[0]; output->height = shape[1];
        output->width = shape[2];
    } else if (dims == 4 && shape[2] == expected_channels) {
        output->layout = YOLOV8N_LAYOUT_NHCW;
        output->height = shape[1]; output->channels = shape[2];
        output->width = shape[3];
    } else if (dims == 3 && shape[1] == expected_channels) {
        output->layout = YOLOV8N_LAYOUT_NHCW;
        output->height = shape[0]; output->channels = shape[1];
        output->width = shape[2];
    } else {
        return -1;
    }
    if (model_get_output_datatype(model, index) != VBX_CNN_CALC_TYPE_INT8) {
        return -1;
    }
    output->data = data;
    output->scale =
        (float)model_get_output_scale_fix16_value(model, index) / 65536.0f;
    output->zero_point = model_get_output_zeropoint(model, index);
    layout_name = output->layout == YOLOV8N_LAYOUT_NHWC ? "NHWC" :
                  output->layout == YOLOV8N_LAYOUT_NHCW ? "NHCW" : "NCHW";
    printf("output[%d]: %s 1x%dx%dx%d scale=%g zero=%d\n", index,
           layout_name, output->height, output->width, output->channels,
           output->scale, output->zero_point);
    return output->scale > 0.0f ? 0 : -1;
}

int yolo_init(const char *model_path, float confidence_threshold,
              float nms_iou_threshold)
{
    unsigned output_index;
    size_t input_bytes;

    if (model_path == NULL || confidence_threshold < 0.0f ||
        confidence_threshold > 1.0f || nms_iou_threshold < 0.0f ||
        nms_iou_threshold > 1.0f) return -1;
    if (g_yolo.model != NULL) return 0;
    if (check_board_devices() != 0) return -2;

    memset(&g_yolo, 0, sizeof(g_yolo));
    g_yolo.vbx = vbx_cnn_init(NULL);
    if (g_yolo.vbx == NULL) return -3;
    g_yolo.model = load_model_to_dma(g_yolo.vbx, model_path);
    if (g_yolo.model == NULL) return -4;

    g_yolo.input_count = model_get_num_inputs(g_yolo.model);
    g_yolo.output_count = model_get_num_outputs(g_yolo.model);
    if (g_yolo.input_count != EXPECTED_INPUTS ||
        g_yolo.output_count != YOLOV8N_NUM_OUTPUTS ||
        g_yolo.input_count + g_yolo.output_count > MAX_IO_BUFFERS) return -5;
    if (get_input_geometry(g_yolo.model, &g_yolo.input_width,
                           &g_yolo.input_height, &g_yolo.input_is_hcw) != 0) {
        return -6;
    }
    if (model_get_input_datatype(g_yolo.model, 0) !=
            VBX_CNN_CALC_TYPE_UINT8 ||
        model_get_input_zeropoint(g_yolo.model, 0) != 0) return -7;

    g_yolo.output_channels =
        infer_output_channels(g_yolo.model, g_yolo.output_count);
    g_yolo.num_classes =
        g_yolo.output_channels - YOLOV8N_DFL_CHANNELS;
    if (g_yolo.num_classes <= 0) return -8;

    input_bytes = model_get_input_length(g_yolo.model, 0) * sizeof(uint8_t);
    if (input_bytes != (size_t)g_yolo.input_width *
                       (size_t)g_yolo.input_height * 3U) return -9;
    g_yolo.io[0] = (vbx_cnn_io_ptr_t)vbx_allocate_dma_buffer(
        g_yolo.vbx, input_bytes, 0);
    if (g_yolo.io[0] == 0) return -10;

    for (output_index = 0; output_index < g_yolo.output_count;
         ++output_index) {
        size_t output_bytes =
            model_get_output_length(g_yolo.model, output_index) *
            sizeof(int8_t);
        g_yolo.io[g_yolo.input_count + output_index] =
            (vbx_cnn_io_ptr_t)vbx_allocate_dma_buffer(
                g_yolo.vbx, output_bytes, 0);
        if (g_yolo.io[g_yolo.input_count + output_index] == 0) return -11;
    }
#if USE_INTERRUPTS
    {
        uint32_t enable = 1;
        if (write(g_yolo.vbx->fd, &enable, sizeof(enable)) < 0) return -12;
    }
#endif
    g_yolo.confidence_threshold = confidence_threshold;
    g_yolo.nms_iou_threshold = nms_iou_threshold;
    printf("YOLO initialized: input=%dx%d classes=%d conf=%.3f nms=%.3f\n",
           g_yolo.input_width, g_yolo.input_height, g_yolo.num_classes,
           g_yolo.confidence_threshold, g_yolo.nms_iou_threshold);
    return 0;
}

int yolo_detect(const uint8_t *pixels, int width, int height,
                int stride_bytes, int channels,
                yolov8n_detection_t *detections, int max_detections)
{
    const uint8_t *source = pixels;
    uint8_t *packed = NULL;
    uint8_t *input = NULL;
    yolov8n_config_t config;
    unsigned output_index;
    int status;

    if (g_yolo.model == NULL || pixels == NULL || detections == NULL ||
        max_detections <= 0 || width <= 0 || height <= 0 || channels != 3 ||
        stride_bytes < width * channels) return -1;

    if (stride_bytes != width * channels) {
        int y;
        packed = (uint8_t *)malloc((size_t)width * (size_t)height * 3U);
        if (packed == NULL) return -2;
        for (y = 0; y < height; ++y) {
            memcpy(packed + (size_t)y * (size_t)width * 3U,
                   pixels + (size_t)y * (size_t)stride_bytes,
                   (size_t)width * 3U);
        }
        source = packed;
    }
    input = preprocess_rgb_planar(source, width, height,
                                  g_yolo.input_width, g_yolo.input_height);
    free(packed);
    if (input == NULL) return -3;
    if (g_yolo.input_is_hcw) {
        input = repack_chw_to_hcw(input, g_yolo.input_width,
                                  g_yolo.input_height);
        if (input == NULL) return -4;
    }
    memcpy((void *)g_yolo.io[0], input,
           model_get_input_length(g_yolo.model, 0));
    free(input);

    status = vbx_cnn_model_start(g_yolo.vbx, g_yolo.model, g_yolo.io);
    if (status >= 0) {
#if USE_INTERRUPTS
        status = vbx_cnn_model_wfi(g_yolo.vbx);
#else
        while ((status = vbx_cnn_model_poll(g_yolo.vbx)) > 0) {
            /* DMA transfer and accelerator wait are inside inference. */
        }
#endif
    }
    if (status < 0) return -5;

    for (output_index = 0; output_index < g_yolo.output_count;
         ++output_index) {
        if (configure_output(
                g_yolo.model, (int)output_index,
                (const int8_t *)g_yolo.io[g_yolo.input_count + output_index],
                g_yolo.output_channels, &g_yolo.outputs[output_index]) != 0) {
            return -6;
        }
    }
    memset(&config, 0, sizeof(config));
    config.input_width = g_yolo.input_width;
    config.input_height = g_yolo.input_height;
    config.image_width = width;
    config.image_height = height;
    config.num_classes = g_yolo.num_classes;
    config.transpose_xy = g_yolo.input_is_hcw;
    config.confidence_threshold = g_yolo.confidence_threshold;
    config.nms_iou_threshold = g_yolo.nms_iou_threshold;
    return yolov8n_postprocess(g_yolo.outputs, &config,
                               detections, max_detections);
}

#ifndef YOLOV8N_LIBRARY_ONLY
int main(int argc, char **argv)
{
    unsigned char *rgb = NULL;
    yolov8n_detection_t detections[MAX_STANDALONE_DETECTIONS];
    float confidence = 0.30f;
    float nms = 0.40f;
    struct timespec start;
    struct timespec end;
    int width = 0;
    int height = 0;
    int count;
    int status;
    int i;

    if (argc < 3 || argc > 5) {
        fprintf(stderr, "Usage: %s MODEL.vnnx IMAGE.jpg [CONF] [NMS]\n",
                argv[0]);
        return 2;
    }
    if (argc >= 4) confidence = strtof(argv[3], NULL);
    if (argc >= 5) nms = strtof(argv[4], NULL);

    status = yolo_init(argv[1], confidence, nms);
    if (status != 0) {
        fprintf(stderr, "YOLO initialization failed: %d\n", status);
        return 1;
    }
    if (!read_JPEG_file(argv[2], &width, &height, &rgb, 0) || rgb == NULL) {
        fprintf(stderr, "Unable to read JPEG: %s\n", argv[2]);
        free(rgb);
        return 1;
    }
    printf("JPEG input: %dx%dx3 RGB HWC, stride=%d\n",
           width, height, width * 3);

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    count = yolo_detect(rgb, width, height, width * 3, 3,
                        detections, MAX_STANDALONE_DETECTIONS);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    free(rgb);
    if (count < 0) {
        fprintf(stderr, "YOLO detection failed: %d\n", count);
        return 1;
    }
    printf("YOLO preprocess+DMA inference+postprocess: %.3f ms\n",
           elapsed_ms(&start, &end));
    printf("detections: %d\n", count);
    for (i = 0; i < count; ++i) {
        printf("class=%d %.4f (%.1f, %.1f, %.1f, %.1f)\n",
               detections[i].class_id, detections[i].confidence,
               detections[i].x1, detections[i].y1,
               detections[i].x2, detections[i].y2);
    }
    return 0;
}
#endif
