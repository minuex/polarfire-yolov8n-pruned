/*
 * pipeline_main.c — the acquisition/tracking state machine, laid out to match
 * the integration flow diagram.
 *
 * [1] INITIALISATION            camera model, target model, image parameters
 *            |
 *            v
 * [2] ACQUISITION               image -> bbox -> segments -> features -> pose
 *            |  initial pose
 *            v
 * [3] PREVIOUS POSE  <---------------------+   (tracking prior)
 *            |  prior                      |
 *            v                             |
 * [4] TRACKING                             |  yes
 *            |  next frame, predicted ROI  |
 *            v                             |
 * [5] TRACKING SUCCESS?  -------------------+
 *            |  no
 *            v
 *     back to [2] ACQUISITION
 *
 * Each box is tagged [n] in the code below. [2] and [4] run the same four
 * stages and differ only in which solver is called.
 *
 * acquire_frame()은 명령행으로 전달된 2048x2048 JPEG를 읽는다.
 * acquire_rois()는 외부 YOLO 추론 연결 경계에 연결되어 있다.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "Helpers.h"
#include "PoseDetParams.h"
#include "demo_common.h"
#include "demo_debug.h"
#include "model_embedded.h"
#include "pose_api.h"
#include "postprocess_yolov8n.h"

#ifndef DEMO_DATA_DIR
#define DEMO_DATA_DIR "data"
#endif

/* PolarFire 최초 통합 시험: JPEG 한 장으로 acquisition 한 프레임만 실행한다. */
#define FIRST_FRAME 0
#define LAST_FRAME  0
#define MAX_ROIS    8
#define MAX_YOLO_DETECTIONS 100
#define YOLO_BBOX_EXPANSION 0.10f
#define PIPELINE_IMAGE_WIDTH  2048
#define PIPELINE_IMAGE_HEIGHT 2048

/* Width of the result column in the printed table, so every row lines up. */
#define RESULT_W    34

/* [5] lost-lock bounds: the tracked pose must stay within reach of the prior.
 * Statistical rejection of a measurement belongs to the navigation filter. */
#define SUCCESS_MAX_DQ_DEG      30.0f   /* pose jump bound, attitude    */
#define SUCCESS_MAX_DT_M        1.0f    /* pose jump bound, translation */

/* pose_acquire()/pose_track()에 전달할 원본 이미지 한 프레임의 형식이다. */
typedef struct {
    const uint8_t *pixels;  /* 원본 이미지의 첫 픽셀, 채널당 8비트             */
    int width, height;
    int stride_bytes;       /* 한 행의 바이트 수. packed이면 width*channels    */
    int channels;           /* 1=grayscale, 3=RGB HWC interleaved             */
    void *owner;            /* release_frame()에서 해제할 메모리 포인터         */
} PipelineFrame;

/* VectorBlox SDK example/postprocess/image.c의 JPEG 함수(-ljpeg 링크). */
extern int read_JPEG_file(const char *filename,
                          int *width,
                          int *height,
                          unsigned char **image,
                          const int grayscale);

// acquisition/tracking 루프 시작 전에 argv[1]의 JPEG 경로를 저장한다. 
static const char *g_jpeg_path = NULL;

/* ==========================================================================
 * JPEG 이미지 입력
 *
 * 입력:
 *   frame_index        int                    가져올 프레임 번호
 * 출력:
 *   out->pixels        const uint8_t *        원본 이미지의 첫 픽셀
 *   out->width         int                    이미지 너비(픽셀)
 *   out->height        int                    이미지 높이(픽셀)
 *   out->stride_bytes  int                    한 행의 바이트 수
 *   out->channels      int                    RGB HWC이므로 3
 *   out->owner         void *                 release_frame()에서 해제할 버퍼
 * 반환값: 성공하면 1, 이미지를 읽지 못하면 0
 *
 * 전달받은 read_JPEG_file() 사용 예:
 *
 *     unsigned char *img = NULL;  int w = 0, h = 0;
 *     if (!read_JPEG_file(path, &w, &h, &img, 0) || !img) return 0;
 *     out->pixels = img;  out->width = w;  out->height = h;
 *     out->stride_bytes = w * 3;  out->channels = 3;  out->owner = img;
 *
 * grayscale=0을 전달하면 RGB로 디코딩한다. malloc으로 생성된 버퍼의
 * 소유권은 PipelineFrame이 가지며 pose_acquire()/pose_track() 완료 후 해제한다.
 * ========================================================================== */
static int acquire_frame(int frame_index, PipelineFrame *out)
{
    unsigned char *rgb = NULL;
    int width = 0;
    int height = 0;

    (void)frame_index; // 현재 시험에서는 동일한 JPEG 한 장을 사용
    if (out == NULL || g_jpeg_path == NULL) return 0;
    memset(out, 0, sizeof(*out));

    if (!read_JPEG_file(g_jpeg_path, &width, &height, &rgb, 0) || rgb == NULL) {
        free(rgb);
        return 0;
    }
    if (width != PIPELINE_IMAGE_WIDTH || height != PIPELINE_IMAGE_HEIGHT) {
        fprintf(stderr, "Expected %dx%d JPEG, got %dx%d: %s\n",
                PIPELINE_IMAGE_WIDTH, PIPELINE_IMAGE_HEIGHT,
                width, height, g_jpeg_path);
        free(rgb);
        return 0;
    }

    // 출력 형식: uint8 RGB, HWC interleaved, 행 패딩 없는 packed 배열
    out->pixels = rgb;
    out->width = width;
    out->height = height;
    out->stride_bytes = width * 3; 
    out->channels = 3;
    out->owner = rgb;
    return 1;
}

static void release_frame(PipelineFrame *f)
{
    if (f->owner) {
        // read_JPEG_file()이 malloc으로 할당한 원본 JPEG 디코딩 버퍼 
        free(f->owner);
        f->owner = NULL;
    }
    f->pixels = NULL;
}

/* ==========================================================================
 * YOLO BBOX -> POSE ROI 연결부
 *
 * yolo_detect()는 기존에 검증된 VNNX 추론/후처리 경로를 실행한다.
 * 이 함수는 원본 이미지 좌표의 bbox를 pose_roi_input_t로 변환한다.
 *
 * 입력:
 *   frame          const PipelineFrame *     원본 RGB 이미지와 크기 정보
 *   max_rois       int                       rois 배열의 최대 용량
 * 출력:
 *   rois           pose_roi_input_t[max_rois]
 *       x1,y1은 좌상단, x2,y2는 우하단의 inclusive 원본 이미지 좌표이다.
 *       width=x2-x1+1, height=y2-y1+1이며 confidence와 class_id도 전달한다.
 *       최종 배열은 confidence 내림차순이다.
 * 반환값: 생성한 ROI 개수, 검출 결과가 없으면 0
 *
 * 클래스별 confidence가 가장 높은 bbox 하나만 남긴다(클래스별 Top-1).
 * 선택된 bbox는 전체 폭과 높이를 각각 10% 확장한다(각 방향으로 5%).
 * 확장 좌표는 원본 이미지 경계로 제한하며 ROI 픽셀을 여기서 crop하지 않는다.
 * pose_acquire()/pose_track()에는 원본 이미지 버퍼와 ROI 좌표를 함께 넘긴다.
 * 실제 포즈 계산은 cfg.target_class_id와 일치하는 ROI에 대해서만 수행된다.
 * ========================================================================== */
static int clamp_int(int value, int low, int high)
{
    if (value < low) return low;
    if (value > high) return high;
    return value;
}

static int compare_pose_rois(const void *left, const void *right)
{
    const pose_roi_input_t *a = (const pose_roi_input_t *)left;
    const pose_roi_input_t *b = (const pose_roi_input_t *)right;
    if (a->confidence < b->confidence) return 1;
    if (a->confidence > b->confidence) return -1;
    return 0;
}

// 원본 이미지 좌표의 YOLO bbox 하나를 10% 확장된 Pose ROI로 변환
static int yolo_box_to_pose_roi(const yolov8n_detection_t *d,
                                int image_width,
                                int image_height,
                                pose_roi_input_t *roi)
{
    const float width = d->x2 - d->x1;
    const float height = d->y2 - d->y1;
    const float margin_x = 0.5f * YOLO_BBOX_EXPANSION * width;
    const float margin_y = 0.5f * YOLO_BBOX_EXPANSION * height;

    if (!isfinite(d->x1) || !isfinite(d->y1) ||
        !isfinite(d->x2) || !isfinite(d->y2) ||
        !isfinite(d->confidence) || width <= 0.0f || height <= 0.0f) {
        return 0;
    }

    roi->x1 = clamp_int((int)floorf(d->x1 - margin_x), 0, image_width - 1);
    roi->y1 = clamp_int((int)floorf(d->y1 - margin_y), 0, image_height - 1);
    roi->x2 = clamp_int((int)ceilf(d->x2 + margin_x), 0, image_width - 1);
    roi->y2 = clamp_int((int)ceilf(d->y2 + margin_y), 0, image_height - 1);
    if (roi->x2 < roi->x1 || roi->y2 < roi->y1) return 0;

    roi->width = roi->x2 - roi->x1 + 1;
    roi->height = roi->y2 - roi->y1 + 1;
    roi->confidence = d->confidence;
    roi->class_id = d->class_id;
    return 1;
}

static int acquire_rois(const PipelineFrame *frame,
                        pose_roi_input_t *rois,
                        int max_rois)
{
    yolov8n_detection_t detections[MAX_YOLO_DETECTIONS];
    int detection_count;
    int roi_count = 0;
    int i;

    if (frame == NULL || frame->pixels == NULL || rois == NULL ||
        max_rois <= 0 || max_rois > MAX_ROIS) {
        return 0;
    }

    // 기존 VNNX 추론 및 yolov8n_postprocess() 구현을 호출 
    detection_count = yolo_detect(frame->pixels,
                                  frame->width,
                                  frame->height,
                                  frame->stride_bytes,
                                  frame->channels,
                                  detections,
                                  MAX_YOLO_DETECTIONS);
    if (detection_count <= 0) return 0;
    if (detection_count > MAX_YOLO_DETECTIONS) {
        detection_count = MAX_YOLO_DETECTIONS;
    }

    // 클래스별 Top-1: 모델의 클래스 수는 유동적으로 받아들이게끔
    for (i = 0; i < detection_count; ++i) {
        pose_roi_input_t candidate;
        int same_class = -1;
        int k;

        if (!yolo_box_to_pose_roi(&detections[i], frame->width,
                                  frame->height, &candidate)) {
            continue;
        }
        for (k = 0; k < roi_count; ++k) {
            if (rois[k].class_id == candidate.class_id) {
                same_class = k;
                break;
            }
        }
        if (same_class >= 0) {
            if (candidate.confidence > rois[same_class].confidence) {
                rois[same_class] = candidate;
            }
        } else if (roi_count < max_rois) {
            rois[roi_count++] = candidate;
        }
    }

    qsort(rois, (size_t)roi_count, sizeof(rois[0]), compare_pose_rois);
    return roi_count;
}

/* ========================================================================== */

typedef enum { STATE_ACQUISITION, STATE_TRACKING } pipeline_state_t;

static float attitude_delta_deg(const float a[4], const float b[4])
{
    float d = fabsf(a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]);
    if (d > 1.0f) d = 1.0f;
    return 2.0f * acosf(d) * 180.0f / (float)M_PI;
}

/* 1 when the tracked pose is usable; `why` gets a short reason when it is not. */
static int tracking_succeeded(const pose_result_t *r,
                              const float q_prev[4], const float t_prev[3],
                              const char **why)
{
    if (r->status != POSE_STATUS_OK) { *why = "solver rejected the frame"; return 0; }

    const float dq = attitude_delta_deg(r->q, q_prev);
    const float dt = sqrtf((r->t[0]-t_prev[0])*(r->t[0]-t_prev[0]) +
                           (r->t[1]-t_prev[1])*(r->t[1]-t_prev[1]) +
                           (r->t[2]-t_prev[2])*(r->t[2]-t_prev[2]));
    if (dq > SUCCESS_MAX_DQ_DEG) { *why = "attitude jump over bound"; return 0; }
    if (dt > SUCCESS_MAX_DT_M)   { *why = "translation jump over bound"; return 0; }

    *why = NULL;
    return 1;
}

int main(int argc, char **argv)
{
    /* ======================================================================
     * [1] INITIALISATION — camera model, target model, image parameters.
     * ====================================================================== */
    pose_config_t cfg;
    float confidence_threshold;
    float nms_iou_threshold;
    int yolo_status;

    if (argc < 3 || argc > 5) {
        fprintf(stderr,
                "Usage: %s MODEL.vnnx IMAGE_2048.jpg [CONF] [NMS]\n",
                argv[0]);
        return 1;
    }
    g_jpeg_path = argv[2];
    confidence_threshold = argc > 3 ? strtof(argv[3], NULL) : 0.30f;
    nms_iou_threshold = argc > 4 ? strtof(argv[4], NULL) : 0.40f;
    yolo_status = yolo_init(argv[1], confidence_threshold, nms_iou_threshold);
    if (yolo_status != 0) {
        fprintf(stderr, "YOLO initialization failed: %d\n", yolo_status);
        return 1;
    }
    pose_config_init(&cfg);              /* target model: compiled in         */
    demo_camera_matrix(cfg.K);           /* replace with your calibration     */
    demo_distortion_coeffs(cfg.dist_coeffs);
    /* The bundled set is rendered undistorted, so demo_distortion_coeffs() are
     * all zero and this flag costs nothing; a real camera sets both. */
    cfg.use_distortion = 1;
    cfg.target_class_id = 0;

    /* Validation set only: used to print the error columns. */
    DemoPose gt[DEMO_MAX_FRAMES];
    const int have_gt =
        demo_load_ground_truth(DEMO_DATA_DIR "/tracking/truth.csv", gt) > 0;

    pipeline_state_t state = STATE_ACQUISITION;
    float q_prev[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
    float t_prev[3] = { 0.0f, 0.0f, 0.0f };
    int n_acq = 0, n_track = 0, n_lost = 0;

    demo_dbg_msg("frame  state        %-*s %8s  %s\n",
                 RESULT_W, "result", "segments", have_gt ? "att_err     pos_err" : "");
    demo_dbg_msg("%s\n", "------------------------------------------------------------------------------------");

    for (int i = FIRST_FRAME; i <= LAST_FRAME; ++i) {
        const char *state_name =
            (state == STATE_TRACKING) ? "TRACKING" : "ACQUISITION";

        /* ---- [2]/[4] stage 1: IMAGE ACQUISITION ---- */
        PipelineFrame frame;
        if (!acquire_frame(i, &frame)) {
            demo_dbg_msg("%5d  %-11s  no frame\n", i, state_name);
            continue;
        }

        /* ---- stage 2: BBOX PROCESSING ---- */
        pose_roi_input_t rois[MAX_ROIS];
        int roi_count = acquire_rois(&frame, rois, MAX_ROIS);
        /* 외부 검출 함수의 반환값이므로 배열 범위를 넘지 않도록 다시 검사한다. */
        if (roi_count < 0) roi_count = 0;
        if (roi_count > MAX_ROIS) roi_count = MAX_ROIS;
        if (roi_count == 0) {
            /* No box, so the next frame starts cold. */
            demo_dbg_msg("%5d  %-11s  no detection\n", i, state_name);
            release_frame(&frame);
            state = STATE_ACQUISITION;
            continue;
        }

        /* ---- [3] PREVIOUS POSE — pose_prior_t[roi_count], one copy per ROI:
         * float q[4] (JPL scalar-last) and float t[3] (metres, camera frame).
         * This demo reuses its own previous estimate. INTEGRATION: take the
         * prior from the EKF instead; nothing else in this loop changes. ---- */
        pose_prior_t priors[MAX_ROIS];
        if (state == STATE_TRACKING) {
            for (int k = 0; k < roi_count; ++k) {
                memcpy(priors[k].q, q_prev, sizeof(priors[k].q));
                memcpy(priors[k].t, t_prev, sizeof(priors[k].t));
            }
        }

        /* ---- stages 3 and 4: LINE SEGMENTS + FEATURES, then POSE ----
         * One call crops each ROI, detects segments, maps the endpoints back to
         * full-image coordinates and solves.
         * In:  frame.pixels  const uint8_t *            height x stride_bytes
         *      rois          pose_roi_input_t[roi_count]
         *      priors        pose_prior_t[roi_count]    tracking only;
         *                                              float q[4], float t[3]
         * Out: results       pose_result_t[roi_count]   same order as rois;
         *                                              float q[4], float t[3],
         *                                              int class_id, status,
         *                                              n_segments
         * Returns int: entries written, negative on bad arguments. ---- */
        pose_result_t results[MAX_ROIS];
        int n;
        if (state == STATE_ACQUISITION) {
            n = pose_acquire(&cfg,
                             frame.pixels, frame.width, frame.height,
                             frame.stride_bytes, frame.channels,
                             rois, roi_count, results, MAX_ROIS);
        } else {
            n = pose_track(&cfg,
                           frame.pixels, frame.width, frame.height,
                           frame.stride_bytes, frame.channels,
                           rois, priors, roi_count, results, MAX_ROIS);
        }
        release_frame(&frame);
        if (n < 1) {
            demo_dbg_msg("%5d  %-11s  call failed\n", i, state_name);
            state = STATE_ACQUISITION;
            continue;
        }

        /* Confidence-descending, so the first result of the target class wins. */
        const pose_result_t *r = NULL;
        for (int k = 0; k < n; ++k)
            if (results[k].status != POSE_STATUS_CLASS_SKIP) { r = &results[k]; break; }
        if (r == NULL) {
            demo_dbg_msg("%5d  %-11s  no ROI of the target class\n", i, state_name);
            state = STATE_ACQUISITION;
            continue;
        }

        char err_txt[48] = "";
        if (have_gt && gt[i].valid && r->status == POSE_STATUS_OK) {
            const float dx = r->t[0] - gt[i].t[0];
            const float dy = r->t[1] - gt[i].t[1];
            const float dz = r->t[2] - gt[i].t[2];
            snprintf(err_txt, sizeof(err_txt), "%7.3f deg  %6.3f m",
                     demo_attitude_error_deg(r->q, gt[i].q),
                     sqrtf(dx*dx + dy*dy + dz*dz));
        }

        if (state == STATE_ACQUISITION) {
            /* ---- [2] result -> INITIAL POSE, then on to [3]/[4] ---- */
            ++n_acq;
            if (r->status != POSE_STATUS_OK) {
                demo_dbg_msg("%5d  ACQUISITION  failed, retry next frame\n", i);
                continue;
            }
            memcpy(q_prev, r->q, sizeof(q_prev));
            memcpy(t_prev, r->t, sizeof(t_prev));
            state = STATE_TRACKING;
            demo_dbg_msg("%5d  ACQUISITION  %-*s %8d  %s\n",
                         i, RESULT_W, "acquired -> tracking", r->n_segments, err_txt);
            continue;
        }

        /* ---- [5] TRACKING SUCCESS? ---- */
        ++n_track;
        const char *why = NULL;
        if (tracking_succeeded(r, q_prev, t_prev, &why)) {
            memcpy(q_prev, r->q, sizeof(q_prev));   /* the next [3] PREVIOUS POSE */
            memcpy(t_prev, r->t, sizeof(t_prev));
            demo_dbg_msg("%5d  TRACKING     %-*s %8d  %s\n",
                         i, RESULT_W, "ok", r->n_segments, err_txt);
        } else {
            ++n_lost;
            state = STATE_ACQUISITION;
            char lost[RESULT_W + 1];
            snprintf(lost, sizeof(lost), "LOST (%s)", why);
            demo_dbg_msg("%5d  TRACKING     %-*s %8d\n",
                         i, RESULT_W, lost, r->n_segments);
        }
    }

    demo_dbg_msg("\n%d acquisitions, %d tracked frames, %d lost locks\n",
           n_acq, n_track, n_lost);
    return 0;
}
