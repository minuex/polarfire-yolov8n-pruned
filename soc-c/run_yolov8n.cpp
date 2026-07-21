#include "postprocess_yolov8n.h"
#include "vbx_cnn_api.h"

#include <cassert>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <vector>

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

static uint8_t *preprocess_rgb_planar_image(const uint8_t *interleaved,
                                            int width, int height,
                                            int output_width,
                                            int output_height)
{
    uint8_t *planar = NULL;
    uint8_t *resized = NULL;
    int channel;
    int y;
    int x;

    if (!interleaved || width <= 0 || height <= 0) {
        return NULL;
    }
    planar = static_cast<uint8_t *>(
        std::malloc(static_cast<size_t>(width) * height * 3));
    resized = static_cast<uint8_t *>(
        std::malloc(static_cast<size_t>(output_width) * output_height * 3));
    if (!planar || !resized) {
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
    for (channel = 0; channel < 3; ++channel) {
        resize_image(planar + (size_t)channel * width * height,
                     width, height,
                     resized + (size_t)channel * output_width * output_height,
                     output_width, output_height);
    }
    std::free(planar);
    return resized;
}

static const char *path_basename(const char *path)
{
    const char *slash = std::strrchr(path, '/');
    const char *backslash = std::strrchr(path, '\\');
    const char *base = path;
    if (slash && slash + 1 > base) base = slash + 1;
    if (backslash && backslash + 1 > base) base = backslash + 1;
    return base;
}

static int append_predictions_csv(const char *filename, const char *image_path,
                                  const yolov8n_detection_t *detections,
                                  int count)
{
    FILE *file = std::fopen(filename, "a+");
    if (!file) return -1;
    std::fseek(file, 0, SEEK_END);
    if (std::ftell(file) == 0) {
        std::fprintf(file,
                     "image_id,class_id,confidence,x1,y1,x2,y2\n");
    }
    for (int index = 0; index < count; ++index) {
        std::fprintf(file, "%s,%d,%.9g,%.9g,%.9g,%.9g,%.9g\n",
                     path_basename(image_path), detections[index].class_id,
                     detections[index].confidence, detections[index].x1,
                     detections[index].y1, detections[index].x2,
                     detections[index].y2);
    }
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
                            int expected_channels,
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
        std::fprintf(stderr,
                     "Output %d has no %d-channel CHW/HWC/HCW layout\n",
                     index, expected_channels);
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

static double elapsed_ms_monotonic(const timespec &start, const timespec &end)
{
    return (end.tv_sec - start.tv_sec) * 1000.0 +
           (end.tv_nsec - start.tv_nsec) / 1000000.0;
}

static void print_latency_statistics(const std::vector<double> &samples)
{
    std::vector<double> sorted(samples);
    double sum = 0.0;
    double squared_deviation_sum = 0.0;
    for (double sample : samples) sum += sample;
    double mean = sum / samples.size();
    for (double sample : samples) {
        double deviation = sample - mean;
        squared_deviation_sum += deviation * deviation;
    }
    std::sort(sorted.begin(), sorted.end());
    double median = sorted.size() % 2
        ? sorted[sorted.size() / 2]
        : (sorted[sorted.size() / 2 - 1] + sorted[sorted.size() / 2]) / 2.0;
    size_t p95_index = static_cast<size_t>(
        std::ceil(0.95 * static_cast<double>(sorted.size()))) - 1;
    double standard_deviation =
        std::sqrt(squared_deviation_sum / samples.size());

    std::printf(
        "inference-only: runs=%zu mean=%.3f ms stddev=%.3f ms "
        "median=%.3f ms p95=%.3f ms min=%.3f ms max=%.3f ms "
        "fps=%.3f\n",
        samples.size(), mean, standard_deviation, median, sorted[p95_index],
        sorted.front(), sorted.back(), mean > 0.0 ? 1000.0 / mean : 0.0);
}

static long read_peak_rss_kib(void)
{
    FILE *file = std::fopen("/proc/self/status", "r");
    char line[256];
    long value = -1;
    if (!file) return -1;
    while (std::fgets(line, sizeof(line), file)) {
        if (std::sscanf(line, "VmHWM: %ld kB", &value) == 1) break;
    }
    std::fclose(file);
    return value;
}

static size_t read_dma_pool_bytes(const char **device_name)
{
    const char *name = std::getenv("VBX_DMA_DEVICE");
    char path[256];
    FILE *file;
    unsigned long long bytes = 0;
    if (!name || !*name) name = "udmabuf-ddr-nc0";
    if (device_name) *device_name = name;
    std::snprintf(path, sizeof(path),
                  "/sys/class/u-dma-buf/%s/size", name);
    file = std::fopen(path, "r");
    if (file) {
        if (std::fscanf(file, "%llu", &bytes) != 1) bytes = 0;
        std::fclose(file);
    }
    return bytes > 0 ? static_cast<size_t>(bytes) : 32U * 1024U * 1024U;
}

static int infer_output_channels(model_t *model, unsigned output_count)
{
    int *first_shape = model_get_output_shape(model, 0);
    int first_dims = model_get_output_dims(model, 0);
    if (!first_shape || (first_dims != 3 && first_dims != 4)) return -1;

    for (int candidate_index = 0; candidate_index < first_dims;
         ++candidate_index) {
        int candidate = first_shape[candidate_index];
        bool common = candidate > YOLOV8N_DFL_CHANNELS;
        for (unsigned output_index = 1;
             common && output_index < output_count; ++output_index) {
            int *shape = model_get_output_shape(model, output_index);
            int dims = model_get_output_dims(model, output_index);
            bool found = false;
            if (!shape || (dims != 3 && dims != 4)) return -1;
            for (int d = 0; d < dims; ++d) {
                if (shape[d] == candidate) found = true;
            }
            common = found;
        }
        if (common) return candidate;
    }
    return -1;
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
    int output_channels;
    int num_classes;
    bool input_is_hcw = false;
    int status;
    unsigned input_count;
    unsigned output_count;
    size_t model_dma_bytes;
    size_t input_dma_bytes;
    size_t output_dma_bytes = 0;
    size_t dma_pool_bytes;
    const char *dma_device_name;
    timeval preprocess_start;
    timeval preprocess_end;
    timeval image_io_start;
    timeval image_io_end;
    timeval postprocess_start;
    timeval postprocess_end;
    double image_io_ms;
    double preprocess_ms;
    double network_ms = 0.0;
    double postprocess_ms;
    int warmup_runs;
    int measured_runs;
    timespec model_load_start;
    timespec model_load_end;

    if (argc < 3) {
        std::fprintf(stderr,
                     "Usage: %s MODEL.vnnx IMAGE.jpg [CONF] [NMS] "
                     "[WARMUP] [RUNS] [PREDICTIONS.csv]\n",
                     argv[0]);
        return 1;
    }
    warmup_runs = argc > 5 ? std::strtol(argv[5], NULL, 10) : 0;
    measured_runs = argc > 6 ? std::strtol(argv[6], NULL, 10) : 1;
    if (warmup_runs < 0 || measured_runs <= 0) {
        std::fprintf(stderr, "WARMUP must be >= 0 and RUNS must be > 0\n");
        return 1;
    }

    vbx_cnn = vbx_cnn_init(NULL);
    if (!vbx_cnn) {
        std::fprintf(stderr, "Unable to initialize VectorBlox CNN\n");
        return 1;
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &model_load_start);
    model = read_model_file(vbx_cnn, argv[1]);
    clock_gettime(CLOCK_MONOTONIC_RAW, &model_load_end);
    std::printf("model loading: %.3f ms\n",
                elapsed_ms_monotonic(model_load_start, model_load_end));
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
    output_channels = infer_output_channels(model, output_count);
    num_classes = output_channels - YOLOV8N_DFL_CHANNELS;
    if (num_classes <= 0) {
        std::fprintf(stderr,
                     "Unable to infer class count from output tensor shapes\n");
        return 1;
    }
    std::printf("classes: %d (output channels=%d)\n",
                num_classes, output_channels);

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

    gettimeofday(&image_io_start, NULL);
    read_JPEG_file(argv[2], &image_width, &image_height,
                   &original_image, 0);
    gettimeofday(&image_io_end, NULL);
    image_io_ms = elapsed_us(image_io_start, image_io_end) / 1000.0;
    if (!original_image || image_width <= 0 || image_height <= 0) {
        std::fprintf(stderr, "Unable to read image: %s\n", argv[2]);
        return 1;
    }

    gettimeofday(&preprocess_start, NULL);
    image = preprocess_rgb_planar_image(original_image,
                                        image_width, image_height,
                                        input_width, input_height);
    if (!image) {
        std::fprintf(stderr, "Unable to preprocess image: %s\n", argv[2]);
        std::free(original_image);
        return 1;
    }
    std::free(original_image);
    original_image = NULL;
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
    dma_pool_bytes = read_dma_pool_bytes(&dma_device_name);
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
    std::printf("DMA utilization: allocate-only=%.2f%% runtime-plan=%.2f%% "
                "(device=%s pool=%.2f MiB)\n",
                100.0 * model_dma_bytes / dma_pool_bytes,
                100.0 * (model_dma_bytes + input_dma_bytes + output_dma_bytes) /
                    dma_pool_bytes,
                dma_device_name,
                dma_pool_bytes / (1024.0 * 1024.0));

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
        std::vector<double> latency_samples;
        latency_samples.reserve(static_cast<size_t>(measured_runs));
        for (int run = -warmup_runs; run < measured_runs; ++run) {
            timespec start;
            timespec end;
            clock_gettime(CLOCK_MONOTONIC_RAW, &start);
            status = vbx_cnn_model_start(vbx_cnn, model, io_buffers);
            if (status >= 0) {
#if USE_INTERRUPTS
                status = vbx_cnn_model_wfi(vbx_cnn);
#else
                while ((status = vbx_cnn_model_poll(vbx_cnn)) > 0) {}
#endif
            }
            clock_gettime(CLOCK_MONOTONIC_RAW, &end);
            if (status < 0) break;
            if (run >= 0) {
                latency_samples.push_back(elapsed_ms_monotonic(start, end));
            }
        }
        if (!latency_samples.empty()) {
            double sum = 0.0;
            for (double sample : latency_samples) sum += sample;
            network_ms = sum / latency_samples.size();
            std::printf("warm-up runs: %d (excluded)\n", warmup_runs);
            print_latency_statistics(latency_samples);
        }
    }
    if (status < 0) {
        std::fprintf(stderr, "Inference failed: %d\n",
                     vbx_cnn_get_error_val(vbx_cnn));
        return 1;
    }

    for (unsigned o = 0; o < output_count; ++o) {
        if (configure_output(
                model, static_cast<int>(o),
                reinterpret_cast<const int8_t *>(io_buffers[input_count + o]),
                output_channels,
                &outputs[o]) != 0) {
            return 1;
        }
    }

    gettimeofday(&postprocess_start, NULL);
    config.input_width = input_width;
    config.input_height = input_height;
    config.image_width = image_width;
    config.image_height = image_height;
    config.num_classes = num_classes;
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

    gettimeofday(&postprocess_end, NULL);
    postprocess_ms = elapsed_us(postprocess_start, postprocess_end) / 1000.0;

    std::printf("detections: %d\n", status);
    if (argc > 7 && append_predictions_csv(argv[7], argv[2], detections,
                                            status) != 0) {
        std::fprintf(stderr, "Unable to append predictions to %s\n", argv[7]);
        return 1;
    }
    for (int i = 0; i < status; ++i) {
        std::printf("class=%d %.4f (%.1f, %.1f, %.1f, %.1f)\n",
                    detections[i].class_id,
                    detections[i].confidence,
                    detections[i].x1, detections[i].y1,
                    detections[i].x2, detections[i].y2);
    }
    {
        double total_ms = preprocess_ms + network_ms + postprocess_ms;
        double end_to_end_fps = total_ms > 0.0 ? 1000.0 / total_ms : 0.0;
        std::printf("Stage timing: image_io+jpeg_decode=%.3f ms, "
                    "preprocess=%.3f ms, model_mean=%.3f ms, "
                    "postprocess=%.3f ms\n",
                    image_io_ms, preprocess_ms, network_ms, postprocess_ms);
        std::printf("Single-image compute E2E (pre+model_mean+post)="
                    "%.3f ms, fps=%.3f\n",
                    total_ms, end_to_end_fps);
    }
    {
        long peak_rss_kib = read_peak_rss_kib();
        if (peak_rss_kib >= 0) {
            std::printf("process peak RSS (VmHWM): %ld KiB (%.2f MiB)\n",
                        peak_rss_kib, peak_rss_kib / 1024.0);
        }
    }
    return 0;
}
