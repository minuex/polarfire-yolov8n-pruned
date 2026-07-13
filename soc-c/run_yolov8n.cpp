#include "postprocess_yolov8n.h"
#include "vbx_cnn_api.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/time.h>
#include <unistd.h>

extern "C" {
#include <jpeglib.h>
}

extern "C" int read_JPEG_file(const char *filename, int *width, int *height,
                              unsigned char **image, const int grayscale);
extern "C" int resize_image(uint8_t *image_in, int in_w, int in_h,
                            uint8_t *image_out, int out_w, int out_h);

#ifndef USE_INTERRUPTS
#define USE_INTERRUPTS 1
#endif

static model_t *read_model_file(vbx_cnn_t *vbx_cnn, const char *filename)
{
    FILE *file = std::fopen(filename, "rb");
    model_t *model;
    model_t *dma_model;
    long file_size;
    int data_size;
    int allocate_size;

    if (!file) return NULL;
    std::fseek(file, 0, SEEK_END);
    file_size = std::ftell(file);
    std::fseek(file, 0, SEEK_SET);
    if (file_size <= 0) {
        std::fclose(file);
        return NULL;
    }

    model = static_cast<model_t *>(std::malloc(static_cast<size_t>(file_size)));
    if (!model || std::fread(model, 1, static_cast<size_t>(file_size), file) !=
                      static_cast<size_t>(file_size)) {
        std::free(model);
        std::fclose(file);
        return NULL;
    }
    std::fclose(file);

    data_size = model_get_data_bytes(model);
    std::printf("DATA_BYTES     = %d (%.2f MiB)\n", data_size,
                data_size / (1024.0 * 1024.0));
    if (data_size != file_size) {
        std::fprintf(stderr, "VNNX file-size mismatch: file=%ld metadata=%d\n",
                     file_size, data_size);
        std::free(model);
        return NULL;
    }

    allocate_size = model_get_allocate_bytes(model);
    std::printf("ALLOCATE_BYTES = %d (%.2f MiB)\n", allocate_size,
                allocate_size / (1024.0 * 1024.0));
    std::printf("MODEL_SANITY   = %d\n", model_check_sanity(model));
    dma_model = static_cast<model_t *>(
        vbx_allocate_dma_buffer(vbx_cnn, allocate_size, 0));
    if (!dma_model) {
        std::fprintf(stderr,
                     "Unable to allocate %d bytes from the VectorBlox DMA "
                     "buffer for the model\n",
                     allocate_size);
        std::free(model);
        return NULL;
    }
    std::memcpy(dma_model, model, static_cast<size_t>(data_size));
    std::free(model);
    return dma_model;
}

static uint8_t *read_rgb_planar_image(const char *filename, int output_width,
                                       int output_height, int *image_width,
                                       int *image_height,
                                       uint8_t **original_image)
{
    unsigned char *interleaved = NULL;
    uint8_t *planar = NULL;
    uint8_t *resized = NULL;
    int width = 0;
    int height = 0;
    int channel;
    int y;
    int x;

    read_JPEG_file(filename, &width, &height, &interleaved, 0);
    if (!interleaved || width <= 0 || height <= 0) {
        return NULL;
    }
    planar = static_cast<uint8_t *>(
        std::malloc(static_cast<size_t>(width) * height * 3));
    resized = static_cast<uint8_t *>(
        std::malloc(static_cast<size_t>(output_width) * output_height * 3));
    if (!planar || !resized) {
        std::free(interleaved);
        std::free(planar);
        std::free(resized);
        return NULL;
    }

    for (y = 0; y < height; ++y) {
        for (x = 0; x < width; ++x) {
            for (channel = 0; channel < 3; ++channel) {
                planar[(size_t)channel * width * height + (size_t)y * width + x] =
                    interleaved[((size_t)y * width + x) * 3 + channel];
            }
        }
    }
    if (original_image) {
        *original_image = interleaved;
    } else {
        std::free(interleaved);
    }

    for (channel = 0; channel < 3; ++channel) {
        resize_image(planar + (size_t)channel * width * height,
                     width, height,
                     resized + (size_t)channel * output_width * output_height,
                     output_width, output_height);
    }
    std::free(planar);
    *image_width = width;
    *image_height = height;
    return resized;
}

static int clamp_int(int value, int low, int high)
{
    if (value < low) return low;
    if (value > high) return high;
    return value;
}

static void draw_detection_box(uint8_t *rgb, int width, int height,
                               const yolov8n_detection_t *detection)
{
    int x1 = clamp_int(static_cast<int>(detection->x1 + 0.5f), 0, width - 1);
    int y1 = clamp_int(static_cast<int>(detection->y1 + 0.5f), 0, height - 1);
    int x2 = clamp_int(static_cast<int>(detection->x2 + 0.5f), 0, width - 1);
    int y2 = clamp_int(static_cast<int>(detection->y2 + 0.5f), 0, height - 1);
    int thickness = (width < height ? width : height) / 256;
    if (thickness < 2) thickness = 2;

    for (int t = 0; t < thickness; ++t) {
        int top = clamp_int(y1 + t, 0, height - 1);
        int bottom = clamp_int(y2 - t, 0, height - 1);
        int left = clamp_int(x1 + t, 0, width - 1);
        int right = clamp_int(x2 - t, 0, width - 1);
        for (int x = x1; x <= x2; ++x) {
            size_t top_index = (static_cast<size_t>(top) * width + x) * 3;
            size_t bottom_index = (static_cast<size_t>(bottom) * width + x) * 3;
            rgb[top_index] = rgb[bottom_index] = 0;
            rgb[top_index + 1] = rgb[bottom_index + 1] = 255;
            rgb[top_index + 2] = rgb[bottom_index + 2] = 0;
        }
        for (int y = y1; y <= y2; ++y) {
            size_t left_index = (static_cast<size_t>(y) * width + left) * 3;
            size_t right_index = (static_cast<size_t>(y) * width + right) * 3;
            rgb[left_index] = rgb[right_index] = 0;
            rgb[left_index + 1] = rgb[right_index + 1] = 255;
            rgb[left_index + 2] = rgb[right_index + 2] = 0;
        }
    }
}

static int write_rgb_jpeg(const char *filename, const uint8_t *rgb,
                          int width, int height, int quality)
{
    FILE *file = std::fopen(filename, "wb");
    jpeg_compress_struct compressor;
    jpeg_error_mgr error_manager;
    if (!file) return -1;

    compressor.err = jpeg_std_error(&error_manager);
    jpeg_create_compress(&compressor);
    jpeg_stdio_dest(&compressor, file);
    compressor.image_width = width;
    compressor.image_height = height;
    compressor.input_components = 3;
    compressor.in_color_space = JCS_RGB;
    jpeg_set_defaults(&compressor);
    jpeg_set_quality(&compressor, quality, TRUE);
    jpeg_start_compress(&compressor, TRUE);
    while (compressor.next_scanline < compressor.image_height) {
        JSAMPROW row = const_cast<JSAMPROW>(
            rgb + static_cast<size_t>(compressor.next_scanline) * width * 3);
        jpeg_write_scanlines(&compressor, &row, 1);
    }
    jpeg_finish_compress(&compressor);
    jpeg_destroy_compress(&compressor);
    std::fclose(file);
    return 0;
}

static uint8_t *repack_chw_to_hcw(uint8_t *chw, int width, int height)
{
    uint8_t *hcw = static_cast<uint8_t *>(
        std::malloc(static_cast<size_t>(width) * height * 3));
    if (!hcw) {
        std::free(chw);
        return NULL;
    }
    for (int y = 0; y < height; ++y) {
        for (int channel = 0; channel < 3; ++channel) {
            std::memcpy(hcw + ((static_cast<size_t>(y) * 3 + channel) * width),
                        chw + ((static_cast<size_t>(channel) * height + y) * width),
                        static_cast<size_t>(width));
        }
    }
    std::free(chw);
    return hcw;
}

static int configure_output(model_t *model, int index, const int8_t *data,
                            yolov8n_output_t *output)
{
    int *shape = model_get_output_shape(model, index);
    int dims = model_get_output_dims(model, index);
    if (!shape || (dims != 3 && dims != 4)) {
        std::fprintf(stderr, "Output %d must be rank 3 or 4\n", index);
        return -1;
    }

    std::printf("output[%d] raw shape: [", index);
    for (int d = 0; d < dims; ++d) {
        std::printf("%s%d", d ? "," : "", shape[d]);
    }
    std::printf("]\n");

    if (dims == 4 && shape[3] == YOLOV8N_OUTPUT_CHANNELS) {
        output->layout = YOLOV8N_LAYOUT_NHWC;
        output->height = shape[1];
        output->width = shape[2];
        output->channels = shape[3];
    } else if (dims == 4 && shape[1] == YOLOV8N_OUTPUT_CHANNELS) {
        output->layout = YOLOV8N_LAYOUT_NCHW;
        output->channels = shape[1];
        output->height = shape[2];
        output->width = shape[3];
    } else if (dims == 3 && shape[2] == YOLOV8N_OUTPUT_CHANNELS) {
        output->layout = YOLOV8N_LAYOUT_NHWC;
        output->height = shape[0];
        output->width = shape[1];
        output->channels = shape[2];
    } else if (dims == 3 && shape[0] == YOLOV8N_OUTPUT_CHANNELS) {
        output->layout = YOLOV8N_LAYOUT_NCHW;
        output->channels = shape[0];
        output->height = shape[1];
        output->width = shape[2];
    } else if (dims == 4 && shape[2] == YOLOV8N_OUTPUT_CHANNELS) {
        output->layout = YOLOV8N_LAYOUT_NHCW;
        output->height = shape[1];
        output->channels = shape[2];
        output->width = shape[3];
    } else if (dims == 3 && shape[1] == YOLOV8N_OUTPUT_CHANNELS) {
        output->layout = YOLOV8N_LAYOUT_NHCW;
        output->height = shape[0];
        output->channels = shape[1];
        output->width = shape[2];
    } else {
        std::fprintf(stderr, "Output %d has no 65-channel CHW/HWC/HCW layout\n",
                     index);
        return -1;
    }

    output->data = data;
    if (model_get_output_datatype(model, index) != VBX_CNN_CALC_TYPE_INT8) {
        std::fprintf(stderr, "Output %d must be INT8\n", index);
        return -1;
    }
    output->scale = static_cast<float>(
        model_get_output_scale_fix16_value(model, index)) / 65536.0f;
    output->zero_point = model_get_output_zeropoint(model, index);
    const char *layout_name = output->layout == YOLOV8N_LAYOUT_NHWC ? "NHWC" :
                              output->layout == YOLOV8N_LAYOUT_NHCW ? "NHCW" :
                                                                    "NCHW";
    std::printf("output[%d]: %s 1x%dx%dx%d scale=%g zero=%d\n", index,
                layout_name,
                output->height, output->width, output->channels,
                output->scale, output->zero_point);
    return output->scale > 0.0f ? 0 : -1;
}

static long elapsed_us(const timeval &start, const timeval &end)
{
    return (end.tv_sec - start.tv_sec) * 1000000L +
           (end.tv_usec - start.tv_usec);
}

int main(int argc, char **argv)
{
    vbx_cnn_t *vbx_cnn;
    model_t *model;
    vbx_cnn_io_ptr_t io_buffers[MAX_IO_BUFFERS] = {};
    yolov8n_output_t outputs[YOLOV8N_NUM_OUTPUTS] = {};
    yolov8n_detection_t detections[100];
    yolov8n_config_t config;
    uint8_t *image = NULL;
    uint8_t *original_image = NULL;
    int image_width = 0;
    int image_height = 0;
    int *input_shape;
    int input_dims;
    int input_width;
    int input_height;
    bool input_is_hcw = false;
    int status;
    unsigned input_count;
    unsigned output_count;
    size_t model_dma_bytes;
    size_t input_dma_bytes;
    size_t output_dma_bytes = 0;
    timeval preprocess_start;
    timeval preprocess_end;
    timeval postprocess_start;
    timeval postprocess_end;
    double preprocess_ms;
    double network_ms;
    double postprocess_ms;

    if (argc < 3) {
        std::fprintf(stderr, "Usage: %s MODEL.vnnx IMAGE.jpg [CONF] [NMS]\n",
                     argv[0]);
        return 1;
    }

    vbx_cnn = vbx_cnn_init(NULL);
    if (!vbx_cnn) {
        std::fprintf(stderr, "Unable to initialize VectorBlox CNN\n");
        return 1;
    }
    model = read_model_file(vbx_cnn, argv[1]);
    if (!model || model_check_sanity(model) != 0) {
        std::fprintf(stderr, "Unable to load a sane VNNX model: %s\n", argv[1]);
        return 1;
    }

    input_count = model_get_num_inputs(model);
    output_count = model_get_num_outputs(model);
    if (input_count != 1 || output_count != YOLOV8N_NUM_OUTPUTS) {
        std::fprintf(stderr, "Expected 1 input and 3 outputs, got %u and %u\n",
                     input_count, output_count);
        return 1;
    }

    input_shape = model_get_input_shape(model, 0);
    input_dims = model_get_input_dims(model, 0);
    if (!input_shape || (input_dims != 3 && input_dims != 4)) {
        std::fprintf(stderr, "Expected rank-3 CHW or rank-4 NCHW input\n");
        return 1;
    }
    std::printf("input raw shape: [");
    for (int d = 0; d < input_dims; ++d) {
        std::printf("%s%d", d ? "," : "", input_shape[d]);
    }
    std::printf("]\n");
    if (input_dims == 4 && input_shape[0] != 1) {
        std::fprintf(stderr, "Expected input batch size 1\n");
        return 1;
    }
    if (input_shape[input_dims - 3] == 3) {
        input_height = input_shape[input_dims - 2];
        input_width = input_shape[input_dims - 1];
    } else if (input_shape[input_dims - 2] == 3) {
        input_is_hcw = true;
        input_height = input_shape[input_dims - 3];
        input_width = input_shape[input_dims - 1];
    } else {
        std::fprintf(stderr, "Expected a 3-channel CHW or HCW input\n");
        return 1;
    }
    if (model_get_input_datatype(model, 0) != VBX_CNN_CALC_TYPE_UINT8 ||
        model_get_input_zeropoint(model, 0) != 0) {
        std::fprintf(stderr,
                     "Expected UINT8 input with zero-point 0; preprocessing "
                     "would need to be changed for this model\n");
        return 1;
    }
    std::printf("input: %s 3x%dx%d scale=%g zero=%d\n",
                input_is_hcw ? "HCW" : "CHW",
                input_height, input_width,
                static_cast<float>(model_get_input_scale_fix16_value(model, 0)) /
                    65536.0f,
                model_get_input_zeropoint(model, 0));

    gettimeofday(&preprocess_start, NULL);
    image = read_rgb_planar_image(argv[2], input_width, input_height,
                                  &image_width, &image_height,
                                  &original_image);
    if (!image) {
        std::fprintf(stderr, "Unable to read image: %s\n", argv[2]);
        return 1;
    }
    if (input_is_hcw) {
        image = repack_chw_to_hcw(image, input_width, input_height);
        if (!image) {
            std::fprintf(stderr, "Unable to repack input image as HCW\n");
            return 1;
        }
    }
    if (model_get_input_length(model, 0) !=
        static_cast<size_t>(input_width) * input_height * 3U) {
        std::fprintf(stderr, "Unexpected input tensor length\n");
        std::free(image);
        return 1;
    }

    model_dma_bytes = static_cast<size_t>(model_get_allocate_bytes(model));
    input_dma_bytes = static_cast<size_t>(model_get_input_length(model, 0)) *
                      sizeof(uint8_t);
    for (unsigned o = 0; o < output_count; ++o) {
        if (model_get_output_datatype(model, o) != VBX_CNN_CALC_TYPE_INT8) {
            std::fprintf(stderr, "Output %u must be INT8\n", o);
            std::free(image);
            return 1;
        }
        output_dma_bytes +=
            static_cast<size_t>(model_get_output_length(model, o)) *
            sizeof(int8_t);
    }
    std::printf("DMA plan: model=%.2f MiB input=%.2f MiB "
                "outputs=%.2f MiB total=%.2f MiB\n",
                model_dma_bytes / (1024.0 * 1024.0),
                input_dma_bytes / (1024.0 * 1024.0),
                output_dma_bytes / (1024.0 * 1024.0),
                (model_dma_bytes + input_dma_bytes + output_dma_bytes) /
                    (1024.0 * 1024.0));

    io_buffers[0] = reinterpret_cast<vbx_cnn_io_ptr_t>(
        vbx_allocate_dma_buffer(vbx_cnn, input_dma_bytes, 0));
    if (!io_buffers[0]) {
        std::fprintf(stderr, "Unable to allocate input DMA buffer\n");
        std::free(image);
        return 1;
    }
    std::memcpy(reinterpret_cast<void *>(io_buffers[0]), image,
                model_get_input_length(model, 0));
    std::free(image);
    gettimeofday(&preprocess_end, NULL);
    preprocess_ms = elapsed_us(preprocess_start, preprocess_end) / 1000.0;

    for (unsigned o = 0; o < output_count; ++o) {
        const size_t output_bytes =
            static_cast<size_t>(model_get_output_length(model, o)) *
            sizeof(int8_t);
        io_buffers[input_count + o] = reinterpret_cast<vbx_cnn_io_ptr_t>(
            vbx_allocate_dma_buffer(vbx_cnn, output_bytes, 0));
        if (!io_buffers[input_count + o]) {
            std::fprintf(stderr, "Unable to allocate output %u DMA buffer\n", o);
            return 1;
        }
    }

#if USE_INTERRUPTS
    {
        uint32_t enable = 1;
        if (write(vbx_cnn->fd, &enable, sizeof(enable)) < 0) {
            std::fprintf(stderr, "Unable to enable VectorBlox interrupt\n");
            return 1;
        }
    }
#endif

    {
        timeval start;
        timeval end;
        gettimeofday(&start, NULL);
        status = vbx_cnn_model_start(vbx_cnn, model, io_buffers);
        if (status >= 0) {
#if USE_INTERRUPTS
            status = vbx_cnn_model_wfi(vbx_cnn);
#else
            while ((status = vbx_cnn_model_poll(vbx_cnn)) > 0) {}
#endif
        }
        gettimeofday(&end, NULL);
        network_ms = elapsed_us(start, end) / 1000.0;
        std::printf("network took %.3f ms\n", network_ms);
    }
    if (status < 0) {
        std::fprintf(stderr, "Inference failed: %d\n",
                     vbx_cnn_get_error_val(vbx_cnn));
        return 1;
    }

    gettimeofday(&postprocess_start, NULL);
    for (unsigned o = 0; o < output_count; ++o) {
        if (configure_output(
                model, static_cast<int>(o),
                reinterpret_cast<const int8_t *>(io_buffers[input_count + o]),
                &outputs[o]) != 0) {
            return 1;
        }
    }

    config.input_width = input_width;
    config.input_height = input_height;
    config.image_width = image_width;
    config.image_height = image_height;
    config.transpose_xy = input_is_hcw ? 1 : 0;
    config.confidence_threshold = argc > 3 ? std::strtof(argv[3], NULL) : 0.30f;
    config.nms_iou_threshold = argc > 4 ? std::strtof(argv[4], NULL) : 0.40f;

    status = yolov8n_postprocess(outputs, &config, detections,
                                 static_cast<int>(sizeof(detections) /
                                                  sizeof(detections[0])));
    if (status < 0) {
        std::fprintf(stderr, "Post-processing failed: %d\n", status);
        return 1;
    }

    std::printf("detections: %d\n", status);
    for (int i = 0; i < status; ++i) {
        std::printf("object %.4f (%.1f, %.1f, %.1f, %.1f)\n",
                    detections[i].confidence,
                    detections[i].x1, detections[i].y1,
                    detections[i].x2, detections[i].y2);
        draw_detection_box(original_image, image_width, image_height,
                           &detections[i]);
    }
    if (write_rgb_jpeg("annotated_output.jpg", original_image,
                       image_width, image_height, 95) == 0) {
        std::printf("Saved annotated output to annotated_output.jpg\n");
    } else {
        std::fprintf(stderr, "Unable to save annotated_output.jpg\n");
    }
    std::free(original_image);
    gettimeofday(&postprocess_end, NULL);
    postprocess_ms = elapsed_us(postprocess_start, postprocess_end) / 1000.0;
    {
        double total_ms = preprocess_ms + network_ms + postprocess_ms;
        double end_to_end_fps = total_ms > 0.0 ? 1000.0 / total_ms : 0.0;
        std::printf("Timing summary: preprocess=%.3f ms, model=%.3f ms, "
                    "postprocess=%.3f ms, total=%.3f ms, "
                    "fps(end-to-end)=%.3f\n",
                    preprocess_ms, network_ms, postprocess_ms, total_ms,
                    end_to_end_fps);
    }
    return 0;
}
