#include "postprocess_yolov8n.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>

typedef struct {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int class_id;
} candidate_t;

static float clampf(float value, float low, float high)
{
    if (value < low) return low;
    if (value > high) return high;
    return value;
}

static float sigmoidf_stable(float value)
{
    if (value >= 0.0f) {
        float z = expf(-value);
        return 1.0f / (1.0f + z);
    }
    {
        float z = expf(value);
        return z / (1.0f + z);
    }
}

static float tensor_value(const yolov8n_output_t *output,
                          int y, int x, int channel)
{
    size_t index;
    if (output->layout == YOLOV8N_LAYOUT_NHWC) {
        index = ((size_t)y * output->width + x) * output->channels + channel;
    } else if (output->layout == YOLOV8N_LAYOUT_NHCW) {
        index = ((size_t)y * output->channels + channel) * output->width + x;
    } else {
        index = ((size_t)channel * output->height + y) * output->width + x;
    }
    return ((float)output->data[index] - (float)output->zero_point) * output->scale;
}

static float dfl_expectation(const yolov8n_output_t *output,
                             int y, int x, int side)
{
    float values[YOLOV8N_REG_MAX];
    float maximum = -INFINITY;
    float weighted = 0.0f;
    float sum = 0.0f;
    int bin;

    for (bin = 0; bin < YOLOV8N_REG_MAX; ++bin) {
        values[bin] = tensor_value(output, y, x,
                                   side * YOLOV8N_REG_MAX + bin);
        if (values[bin] > maximum) maximum = values[bin];
    }
    for (bin = 0; bin < YOLOV8N_REG_MAX; ++bin) {
        float probability = expf(values[bin] - maximum);
        sum += probability;
        weighted += probability * (float)bin;
    }
    return sum > 0.0f ? weighted / sum : 0.0f;
}

static int compare_candidates(const void *left, const void *right)
{
    const candidate_t *a = (const candidate_t *)left;
    const candidate_t *b = (const candidate_t *)right;
    if (a->score < b->score) return 1;
    if (a->score > b->score) return -1;
    return 0;
}

static float box_iou(const candidate_t *a, const candidate_t *b)
{
    float left = a->x1 > b->x1 ? a->x1 : b->x1;
    float top = a->y1 > b->y1 ? a->y1 : b->y1;
    float right = a->x2 < b->x2 ? a->x2 : b->x2;
    float bottom = a->y2 < b->y2 ? a->y2 : b->y2;
    float width = right - left;
    float height = bottom - top;
    float intersection;
    float area_a;
    float area_b;
    float union_area;

    if (width <= 0.0f || height <= 0.0f) return 0.0f;
    intersection = width * height;
    area_a = (a->x2 - a->x1) * (a->y2 - a->y1);
    area_b = (b->x2 - b->x1) * (b->y2 - b->y1);
    union_area = area_a + area_b - intersection;
    return union_area > 0.0f ? intersection / union_area : 0.0f;
}

int yolov8n_postprocess(const yolov8n_output_t outputs[YOLOV8N_NUM_OUTPUTS],
                        const yolov8n_config_t *config,
                        yolov8n_detection_t *detections,
                        int max_detections)
{
    candidate_t *candidates;
    unsigned char *suppressed;
    size_t capacity = 0;
    size_t count = 0;
    int output_index;
    int result_count = 0;

    if (!outputs || !config || !detections || max_detections <= 0 ||
        config->input_width <= 0 || config->input_height <= 0 ||
        config->image_width <= 0 || config->image_height <= 0 ||
        config->num_classes <= 0 ||
        config->confidence_threshold <= 0.0f ||
        config->confidence_threshold >= 1.0f ||
        config->nms_iou_threshold <= 0.0f ||
        config->nms_iou_threshold >= 1.0f) {
        return -1;
    }

    for (output_index = 0; output_index < YOLOV8N_NUM_OUTPUTS; ++output_index) {
        const yolov8n_output_t *output = &outputs[output_index];
        if (!output->data || output->height <= 0 || output->width <= 0 ||
            output->channels != YOLOV8N_DFL_CHANNELS + config->num_classes ||
            output->scale <= 0.0f) {
            return -1;
        }
        capacity += (size_t)output->height * output->width;
    }

    candidates = (candidate_t *)malloc(capacity * sizeof(*candidates));
    if (!candidates) return -2;

    for (output_index = 0; output_index < YOLOV8N_NUM_OUTPUTS; ++output_index) {
        const yolov8n_output_t *output = &outputs[output_index];
        float stride_x = (float)config->input_width / output->width;
        float stride_y = (float)config->input_height / output->height;
        int y;
        int x;

        for (y = 0; y < output->height; ++y) {
            for (x = 0; x < output->width; ++x) {
                float score = 0.0f;
                int class_id = 0;
                candidate_t *candidate;
                float center_x;
                float center_y;

                for (int class_index = 0;
                     class_index < config->num_classes; ++class_index) {
                    float class_score = sigmoidf_stable(tensor_value(
                        output, y, x, YOLOV8N_DFL_CHANNELS + class_index));
                    if (class_score > score) {
                        score = class_score;
                        class_id = class_index;
                    }
                }
                if (score < config->confidence_threshold) continue;

                candidate = &candidates[count++];
                center_x = ((float)x + 0.5f) * stride_x;
                center_y = ((float)y + 0.5f) * stride_y;
                candidate->x1 = center_x - dfl_expectation(output, y, x, 0) * stride_x;
                candidate->y1 = center_y - dfl_expectation(output, y, x, 1) * stride_y;
                candidate->x2 = center_x + dfl_expectation(output, y, x, 2) * stride_x;
                candidate->y2 = center_y + dfl_expectation(output, y, x, 3) * stride_y;
                candidate->score = score;
                candidate->class_id = class_id;
            }
        }
    }

    if (count == 0) {
        free(candidates);
        return 0;
    }

    qsort(candidates, count, sizeof(*candidates), compare_candidates);
    suppressed = (unsigned char *)calloc(count, sizeof(*suppressed));
    if (!suppressed) {
        free(candidates);
        return -2;
    }

    {
        size_t i;
        float scale_x = (float)config->image_width / config->input_width;
        float scale_y = (float)config->image_height / config->input_height;
        for (i = 0; i < count && result_count < max_detections; ++i) {
            size_t j;
            yolov8n_detection_t *detection;
            float x1;
            float y1;
            float x2;
            float y2;
            if (suppressed[i]) continue;

            detection = &detections[result_count++];
            x1 = config->transpose_xy ? candidates[i].y1 : candidates[i].x1;
            y1 = config->transpose_xy ? candidates[i].x1 : candidates[i].y1;
            x2 = config->transpose_xy ? candidates[i].y2 : candidates[i].x2;
            y2 = config->transpose_xy ? candidates[i].x2 : candidates[i].y2;
            detection->x1 = clampf(x1 * scale_x, 0.0f,
                                    (float)(config->image_width - 1));
            detection->y1 = clampf(y1 * scale_y, 0.0f,
                                    (float)(config->image_height - 1));
            detection->x2 = clampf(x2 * scale_x, 0.0f,
                                    (float)(config->image_width - 1));
            detection->y2 = clampf(y2 * scale_y, 0.0f,
                                    (float)(config->image_height - 1));
            detection->confidence = candidates[i].score;
            detection->class_id = candidates[i].class_id;

            for (j = i + 1; j < count; ++j) {
                if (!suppressed[j] &&
                    candidates[i].class_id == candidates[j].class_id &&
                    box_iou(&candidates[i], &candidates[j]) >
                        config->nms_iou_threshold) {
                    suppressed[j] = 1;
                }
            }
        }
    }

    free(suppressed);
    free(candidates);
    return result_count;
}
