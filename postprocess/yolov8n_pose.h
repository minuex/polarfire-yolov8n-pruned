#ifndef YOLOV8N_POSE_H
#define YOLOV8N_POSE_H

#include <stdint.h>

#include "postprocess_yolov8n.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * [1: 원본 이미지 형식]
 *
 * JPEG decode가 만든 원본 프레임을 복사하지 않고 참조하는 view이다.
 * - dtype  : uint8_t
 * - color  : RGB888
 * - layout : HWC (interleaved RGB)
 * - shape  : [image_height, image_width, 3]
 *
 * 현재 2048x2048 이미지라면 shape=[2048,2048,3], stride_bytes=6144이다.
 * rgb 메모리는 이 구조체가 소유하지 않는다. 포즈추정이 끝날 때까지
 * 호출자가 원본 이미지 버퍼를 해제하거나 덮어쓰면 안 된다.
 */
typedef struct {
    const uint8_t *rgb;
    int width;
    int height;
    int stride_bytes;
} yolov8n_pose_image_view_t;

/*
 * [2: YOLO -> pose 출력 형식]
 *
 * 출력은 고정 shape 텐서가 아니라 yolov8n_pose_roi_t rois[M] 배열이다.
 * M은 검출된 클래스 수에 따라 변하며, 클래스별 Top-1만 남기므로
 * M <= num_classes 이다.
 *
 * 좌표는 1024x1024 YOLO 입력 좌표가 아니라 원본 이미지 좌표이다.
 * x1, y1, x2, y2는 모두 inclusive이며 다음 관계를 만족한다.
 * - width  = x2 - x1 + 1
 * - height = y2 - y1 + 1
 *
 * class_id는 학습 YAML의 클래스 순서와 같고, confidence는 YOLO 점수이다.
 */
typedef struct {
    int x1;
    int y1;
    int x2;
    int y2;
    int width;
    int height;
    float confidence;
    int class_id;
} yolov8n_pose_roi_t;

/*
 * [3: 함수 인자와 정책]
 *
 * detections / detection_count
 *   yolov8n_postprocess() 결과와 실제 detection 개수.
 *
 * image_width / image_height
 *   반드시 원본 이미지 크기(현재 2048x2048)를 전달한다.
 *
 * expansion_ratio
 *   bbox 전체 폭/높이의 증가 비율이다. 0.10f이면 최종 폭과 높이가
 *   각각 1.10배가 되며 중심 기준으로 좌/우/상/하에 각각 5% 추가한다.
 *
 * target_class_id
 *   -1이면 모든 클래스를 고려하고, 0 이상이면 해당 class_id만 고려한다.
 *   어느 class_id가 어떤 객체인지는 학습 YAML과 반드시 대조해야 한다.
 *
 * rois / max_rois
 *   호출자가 준비한 출력 배열과 배열 용량. 멀티클래스 전체 전달 시
 *   max_rois는 최소 num_classes 이상을 권장한다.
 *
 * 고정 정책
 *   confidence가 가장 높은 detection 하나만 클래스별로 남긴다(Top-1).
 *   Top-1 박스를 10% 규칙에 따라 확장하고 원본 이미지 경계로 자른다.
 *   반환 ROI는 confidence 내림차순이다.
 *
 * return
 *   0 이상: 생성된 ROI 개수 M, -1: 잘못된 인자.
 */
int yolov8n_pose_make_rois(const yolov8n_detection_t *detections,
                           int detection_count,
                           int image_width,
                           int image_height,
                           float expansion_ratio,
                           int target_class_id,
                           yolov8n_pose_roi_t *rois,
                           int max_rois);

#ifdef __cplusplus
}
#endif

#endif
