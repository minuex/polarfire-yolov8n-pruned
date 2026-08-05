#include "yolov8n_pose.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>

static int clamp_int(int value, int low, int high)
{
    if (value < low) return low;
    if (value > high) return high;
    return value;
}

static int compare_rois(const void *left, const void *right)
{
    /* 최종 pose ROI를 confidence가 높은 순서로 전달 */
    const yolov8n_pose_roi_t *a = (const yolov8n_pose_roi_t *)left;
    const yolov8n_pose_roi_t *b = (const yolov8n_pose_roi_t *)right;
    if (a->confidence < b->confidence) return 1;
    if (a->confidence > b->confidence) return -1;
    return 0;
}

static int detection_to_roi(const yolov8n_detection_t *detection,
                            int image_width,
                            int image_height,
                            float expansion_ratio,
                            yolov8n_pose_roi_t *roi)
{
    float width;
    float height;
    float margin_x;
    float margin_y;

    if (!isfinite(detection->x1) || !isfinite(detection->y1) ||
        !isfinite(detection->x2) || !isfinite(detection->y2) ||
        !isfinite(detection->confidence) ||
        detection->x2 <= detection->x1 ||
        detection->y2 <= detection->y1) {
        return 0;
    }

    /* 10% bbox 확장 */
    width = detection->x2 - detection->x1;
    height = detection->y2 - detection->y1;
    margin_x = 0.5f * expansion_ratio * width;
    margin_y = 0.5f * expansion_ratio * height;

    roi->x1 = clamp_int((int)floorf(detection->x1 - margin_x),
                        0, image_width - 1);
    roi->y1 = clamp_int((int)floorf(detection->y1 - margin_y),
                        0, image_height - 1);
    roi->x2 = clamp_int((int)ceilf(detection->x2 + margin_x),
                        0, image_width - 1);
    roi->y2 = clamp_int((int)ceilf(detection->y2 + margin_y),
                        0, image_height - 1);
    if (roi->x2 < roi->x1 || roi->y2 < roi->y1) return 0;

    roi->width = roi->x2 - roi->x1 + 1;
    roi->height = roi->y2 - roi->y1 + 1;
    roi->confidence = detection->confidence;
    roi->class_id = detection->class_id;
    return 1;
}

int yolov8n_pose_make_rois(const yolov8n_detection_t *detections,
                           int detection_count,
                           int image_width,
                           int image_height,
                           float expansion_ratio,
                           int target_class_id,
                           yolov8n_pose_roi_t *rois,
                           int max_rois)
{
    int output_count = 0;
    int i;

    if ((!detections && detection_count > 0) || detection_count < 0 ||
        image_width <= 0 || image_height <= 0 || expansion_ratio < 0.0f ||
        target_class_id < -1 || !rois || max_rois <= 0) {
        return -1;
    }

    /* 클래스별 Top-1 */
    for (i = 0; i < detection_count; ++i) {
        const yolov8n_detection_t *detection = &detections[i];
        yolov8n_pose_roi_t candidate;
        int same_class_index = -1;
        int roi_index;

        if (target_class_id >= 0 && detection->class_id != target_class_id) {
            continue;
        }
        if (!detection_to_roi(detection, image_width, image_height,
                              expansion_ratio, &candidate)) {
            continue;
        }

        for (roi_index = 0; roi_index < output_count; ++roi_index) {
            if (rois[roi_index].class_id == candidate.class_id) {
                same_class_index = roi_index;
                break;
            }
        }

        if (same_class_index >= 0) {
            if (candidate.confidence > rois[same_class_index].confidence) {
                rois[same_class_index] = candidate;
            }
            continue;
        }

        if (output_count < max_rois) {
            rois[output_count++] = candidate;
        } else {
            int lowest_index = 0;
            for (roi_index = 1; roi_index < output_count; ++roi_index) {
                if (rois[roi_index].confidence <
                    rois[lowest_index].confidence) {
                    lowest_index = roi_index;
                }
            }
            if (candidate.confidence > rois[lowest_index].confidence) {
                rois[lowest_index] = candidate;
            }
        }
    }

    qsort(rois, (size_t)output_count, sizeof(*rois), compare_rois);

    return output_count;
}
