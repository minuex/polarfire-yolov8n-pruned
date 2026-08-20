#define _POSIX_C_SOURCE 200809L

#include "vbx_cnn_api.h"
#include "postprocess_yolov8n.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define EXPECTED_INPUTS 1U
#define EXPECTED_OUTPUTS 3U
#define MAX_DETECTIONS 100
#define ORIGINAL_IMAGE_WIDTH 2048
#define ORIGINAL_IMAGE_HEIGHT 2048

static double elapsed_ms(const struct timespec *start,
                         const struct timespec *end)
{
    return (double)(end->tv_sec - start->tv_sec) * 1000.0 +
           (double)(end->tv_nsec - start->tv_nsec) / 1000000.0;
}

static uint32_t fnv1a32(const int8_t *data, size_t length)
{
    uint32_t hash = UINT32_C(2166136261);
    size_t i;

    for (i = 0; i < length; ++i) {
        hash ^= (uint8_t)data[i];
        hash *= UINT32_C(16777619);
    }
    return hash;
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

static int get_input_geometry(model_t *model,
                              int *width,
                              int *height,
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

static int configure_output(model_t *model,
                            int index,
                            const int8_t *data,
                            int expected_channels,
                            yolov8n_output_t *output)
{
    int *shape = model_get_output_shape(model, index);
    int dims = model_get_output_dims(model, index);
    int d;
    const char *layout_name;

    if (shape == NULL || output == NULL || (dims != 3 && dims != 4)) return -1;
    printf("output[%d] raw shape: [", index);
    for (d = 0; d < dims; ++d) printf("%s%d", d ? "," : "", shape[d]);
    printf("]\n");

    if (dims == 4 && shape[3] == expected_channels) {
        output->layout = YOLOV8N_LAYOUT_NHWC;
        output->height = shape[1];
        output->width = shape[2];
        output->channels = shape[3];
    } else if (dims == 4 && shape[1] == expected_channels) {
        output->layout = YOLOV8N_LAYOUT_NCHW;
        output->channels = shape[1];
        output->height = shape[2];
        output->width = shape[3];
    } else if (dims == 3 && shape[2] == expected_channels) {
        output->layout = YOLOV8N_LAYOUT_NHWC;
        output->height = shape[0];
        output->width = shape[1];
        output->channels = shape[2];
    } else if (dims == 3 && shape[0] == expected_channels) {
        output->layout = YOLOV8N_LAYOUT_NCHW;
        output->channels = shape[0];
        output->height = shape[1];
        output->width = shape[2];
    } else if (dims == 4 && shape[2] == expected_channels) {
        output->layout = YOLOV8N_LAYOUT_NHCW;
        output->height = shape[1];
        output->channels = shape[2];
        output->width = shape[3];
    } else if (dims == 3 && shape[1] == expected_channels) {
        output->layout = YOLOV8N_LAYOUT_NHCW;
        output->height = shape[0];
        output->channels = shape[1];
        output->width = shape[2];
    } else {
        return -1;
    }

    output->data = data;
    output->scale =
        (float)model_get_output_scale_fix16_value(model, index) / 65536.0f;
    output->zero_point = model_get_output_zeropoint(model, index);
    layout_name = output->layout == YOLOV8N_LAYOUT_NHWC ? "NHWC" :
                  output->layout == YOLOV8N_LAYOUT_NHCW ? "NHCW" : "NCHW";
    printf("output[%d]: %s 1x%dx%dx%d scale=%g zero=%d\n",
           index, layout_name, output->height, output->width,
           output->channels, output->scale, output->zero_point);
    return output->scale > 0.0f ? 0 : -1;
}

/*
 * VNNX 파일은 일반 RAM으로 읽어 헤더를 검사한 뒤, 모델 실행에 필요한
 * allocate_bytes 전체를 DMA pool에서 한 번 할당하고 data_bytes만 복사한다.
 */
static model_t *load_model_to_dma(vbx_cnn_t *vbx, const char *path)
{
    FILE *fp = NULL;
    model_t *host_model = NULL;
    model_t *dma_model = NULL;
    long file_size;
    int data_bytes;
    int allocate_bytes;

    fp = fopen(path, "rb");
    if (fp == NULL) {
        fprintf(stderr, "Unable to open VNNX model: %s\n", path);
        return NULL;
    }
    if (fseek(fp, 0, SEEK_END) != 0 ||
        (file_size = ftell(fp)) <= 0 ||
        fseek(fp, 0, SEEK_SET) != 0) {
        fprintf(stderr, "Unable to determine VNNX file size: %s\n", path);
        fclose(fp);
        return NULL;
    }

    host_model = (model_t *)malloc((size_t)file_size);
    if (host_model == NULL ||
        fread(host_model, 1, (size_t)file_size, fp) != (size_t)file_size) {
        fprintf(stderr, "Unable to read complete VNNX model: %s\n", path);
        free(host_model);
        fclose(fp);
        return NULL;
    }
    fclose(fp);

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
        fprintf(stderr, "Unable to allocate model in DMA pool\n");
        free(host_model);
        return NULL;
    }
    memcpy(dma_model, host_model, (size_t)data_bytes);
    free(host_model);
    return dma_model;
}

int main(int argc, char **argv)
{
    vbx_cnn_t *vbx;
    model_t *model;
    vbx_cnn_io_ptr_t io_buffers[MAX_IO_BUFFERS] = {0};
    unsigned input_count;
    unsigned output_count;
    size_t input_bytes;
    uint8_t *input;
    int input_zero_point;
    int input_width;
    int input_height;
    int input_is_hcw;
    int output_channels;
    int num_classes;
    float confidence_threshold = 0.30f;
    float nms_iou_threshold = 0.40f;
    yolov8n_output_t outputs[YOLOV8N_NUM_OUTPUTS] = {{0}};
    yolov8n_detection_t detections[MAX_DETECTIONS];
    yolov8n_config_t config;
    int detection_count;
    int status;
    unsigned o;
    struct timespec start;
    struct timespec end;

    if (argc < 2 || argc > 4) {
        fprintf(stderr, "Usage: %s MODEL.vnnx [CONF] [NMS]\n", argv[0]);
        return 2;
    }
    if (argc >= 3) confidence_threshold = strtof(argv[2], NULL);
    if (argc >= 4) nms_iou_threshold = strtof(argv[3], NULL);
    if (confidence_threshold <= 0.0f || confidence_threshold >= 1.0f ||
        nms_iou_threshold <= 0.0f || nms_iou_threshold >= 1.0f) {
        fprintf(stderr, "CONF and NMS must be between 0 and 1\n");
        return 2;
    }

    vbx = vbx_cnn_init(NULL);
    if (vbx == NULL) {
        fprintf(stderr, "Unable to initialize VectorBlox CNN\n");
        return 1;
    }

    model = load_model_to_dma(vbx, argv[1]);
    if (model == NULL) return 1;

    input_count = model_get_num_inputs(model);
    output_count = model_get_num_outputs(model);
    if (input_count != EXPECTED_INPUTS || output_count != EXPECTED_OUTPUTS) {
        fprintf(stderr, "Expected 1 input and 3 outputs, got %u and %u\n",
                input_count, output_count);
        return 1;
    }
    if (input_count + output_count > MAX_IO_BUFFERS) {
        fprintf(stderr, "Too many model IO buffers\n");
        return 1;
    }
    if (model_get_input_datatype(model, 0) != VBX_CNN_CALC_TYPE_UINT8) {
        fprintf(stderr, "This example expects a UINT8 YOLO input\n");
        return 1;
    }
    if (get_input_geometry(model, &input_width, &input_height,
                           &input_is_hcw) != 0) {
        fprintf(stderr, "Expected a batch-1 CHW or HCW YOLO input\n");
        return 1;
    }
    output_channels = infer_output_channels(model, output_count);
    num_classes = output_channels - YOLOV8N_DFL_CHANNELS;
    if (num_classes <= 0) {
        fprintf(stderr, "Unable to infer YOLO class count\n");
        return 1;
    }
    printf("classes: %d (output channels=%d)\n",
           num_classes, output_channels);

    input_bytes = model_get_input_length(model, 0) * sizeof(uint8_t);
    input = (uint8_t *)vbx_allocate_dma_buffer(vbx, input_bytes, 0);
    if (input == NULL) {
        fprintf(stderr, "Unable to allocate input DMA buffer\n");
        return 1;
    }
    io_buffers[0] = (vbx_cnn_io_ptr_t)input;

    /* JPEG/resize 없이 zero-point 값의 합성 입력으로 순수 추론부만 시험한다. */
    input_zero_point = model_get_input_zeropoint(model, 0);
    memset(input, (uint8_t)input_zero_point, input_bytes);

    for (o = 0; o < output_count; ++o) {
        size_t output_bytes;

        if (model_get_output_datatype(model, o) != VBX_CNN_CALC_TYPE_INT8) {
            fprintf(stderr, "Output %u is not INT8\n", o);
            return 1;
        }
        output_bytes = model_get_output_length(model, o) * sizeof(int8_t);
        io_buffers[input_count + o] = (vbx_cnn_io_ptr_t)
            vbx_allocate_dma_buffer(vbx, output_bytes, 0);
        if (io_buffers[input_count + o] == 0) {
            fprintf(stderr, "Unable to allocate output %u DMA buffer\n", o);
            return 1;
        }
        memset((void *)io_buffers[input_count + o], 0, output_bytes);
    }

    printf("DMA input bytes = %zu\n", input_bytes);
    printf("Starting VectorBlox YOLO inference...\n");

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    status = vbx_cnn_model_start(vbx, model, io_buffers);
    if (status < 0) {
        fprintf(stderr, "vbx_cnn_model_start failed: %d\n", status);
        return 1;
    }
    while ((status = vbx_cnn_model_poll(vbx)) > 0) {
        /* accelerator completion wait is part of inference latency */
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    if (status < 0) {
        fprintf(stderr, "VectorBlox inference failed: %d\n", status);
        return 1;
    }

    printf("inference-only = %.3f ms\n", elapsed_ms(&start, &end));
    for (o = 0; o < output_count; ++o) {
        size_t output_bytes =
            model_get_output_length(model, o) * sizeof(int8_t);
        const int8_t *output =
            (const int8_t *)io_buffers[input_count + o];

        printf("output[%u]: bytes=%zu fnv1a32=0x%08" PRIx32 "\n",
               o, output_bytes, fnv1a32(output, output_bytes));
        if (configure_output(model, (int)o, output, output_channels,
                             &outputs[o]) != 0) {
            fprintf(stderr, "Unable to configure output %u\n", o);
            return 1;
        }
    }

    memset(&config, 0, sizeof(config));
    config.input_width = input_width;
    config.input_height = input_height;
    config.image_width = ORIGINAL_IMAGE_WIDTH;
    config.image_height = ORIGINAL_IMAGE_HEIGHT;
    config.num_classes = num_classes;
    config.transpose_xy = input_is_hcw;
    config.confidence_threshold = confidence_threshold;
    config.nms_iou_threshold = nms_iou_threshold;

    detection_count = yolov8n_postprocess(
        outputs, &config, detections, MAX_DETECTIONS);
    if (detection_count < 0) {
        fprintf(stderr, "YOLO postprocess failed: %d\n", detection_count);
        return 1;
    }
    printf("detections: %d\n", detection_count);
    {
        int i;
        for (i = 0; i < detection_count; ++i) {
            printf("class=%d %.4f (%.1f, %.1f, %.1f, %.1f)\n",
                   detections[i].class_id, detections[i].confidence,
                   detections[i].x1, detections[i].y1,
                   detections[i].x2, detections[i].y2);
        }
    }

    printf("Hello, YOLO DMA inference!\n");
    return 0;
}
