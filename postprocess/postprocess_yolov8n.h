#ifndef POSTPROCESS_YOLOV8N_H
#define POSTPROCESS_YOLOV8N_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define YOLOV8N_NUM_OUTPUTS 3
#define YOLOV8N_REG_MAX 16
#define YOLOV8N_DFL_CHANNELS (4 * YOLOV8N_REG_MAX)

typedef enum {
    YOLOV8N_LAYOUT_NHWC = 0,
    YOLOV8N_LAYOUT_NCHW = 1,
    YOLOV8N_LAYOUT_NHCW = 2
} yolov8n_layout_t;

typedef struct {
    const int8_t *data;
    int height;
    int width;
    int channels;
    yolov8n_layout_t layout;
    float scale;
    int zero_point;
} yolov8n_output_t;

typedef struct {
    int input_width;
    int input_height;
    int image_width;
    int image_height;
    int num_classes;
    int transpose_xy;
    float confidence_threshold;
    float nms_iou_threshold;
} yolov8n_config_t;

typedef struct {
    float x1;
    float y1;
    float x2;
    float y2;
    float confidence;
    int class_id;
} yolov8n_detection_t;

/*
 * Decodes three raw YOLOv8 tensors containing 64 DFL channels followed by
 * one or more class-logit channels. NHWC, NCHW, and VectorBlox NHCW layouts
 * are supported.
 * Coordinates are returned in image_width x image_height space. The caller
 * owns the output array.
 */
int yolov8n_postprocess(const yolov8n_output_t outputs[YOLOV8N_NUM_OUTPUTS],
                        const yolov8n_config_t *config,
                        yolov8n_detection_t *detections,
                        int max_detections);

#ifdef __cplusplus
}
#endif

#endif
