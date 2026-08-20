PolarFire SoC - 순수 C YOLO DMA 추론 및 후처리 예제
==================================================

패키지 구성
-----------

  yolo_dma_hello.c          순수 C DMA 추론 실행부
  postprocess_yolov8n.c/.h  DFL decode, confidence filtering, NMS
  Makefile                  RISC-V lp64d 크로스 빌드
  yolo-dma-hello            빌드 및 보드 검증 완료 바이너리
  models/                   Pruned+KD single VNNX 테스트 모델
  test_images/              2048x2048 JPEG 테스트 이미지 5장
  SHA256SUMS                전체 패키지 무결성 확인

현재 yolo-dma-hello는 JPEG 전처리를 의도적으로 제외한 DMA 추론부 시험이므로
test_images/의 JPEG를 명령행 인자로 사용하지 않는다. 이미지는 기존 전체 YOLO
경로와 향후 JPEG 전처리 연결 시험을 위한 참고 데이터이다.

목적
----

기존 run_yolov8n.cpp 전체 프로그램과 별도로, YOLO 가속기 추론부와
postprocess_yolov8n.c를 순수 C로 컴파일하여 검증한다.

포함 범위:

  VNNX 파일 읽기 및 sanity 검사
  모델 실행 메모리를 64 MiB VectorBlox DMA pool에 할당
  UINT8 입력 DMA buffer 할당
  INT8 raw output 3개 DMA buffer 할당
  vbx_cnn_model_start()
  vbx_cnn_model_poll() 완료 대기
  raw output checksum 출력
  DFL decode, confidence filtering, NMS
  2048x2048 원본 좌표계의 detection 출력
  "Hello, YOLO DMA inference!" 출력

제외 범위:

  JPEG 파일 읽기
  resize/normalization/HCW 변환
  Pose 연결

현재는 모델 input zero-point 값으로 채운 합성 입력을 사용한다. 따라서 출력된
detection은 정확도 평가용이 아니지만, DMA raw output에서 DFL/NMS/BBox까지
이어지는 YOLO 추론 및 후처리 연결 경로를 검증할 수 있다.


1. 사용 소스
-------------

  yolo_dma_c/yolo_dma_hello.c
  yolo_dma_c/postprocess_yolov8n.c
  yolo_dma_c/postprocess_yolov8n.h
  VectorBlox-SDK/drivers/vectorblox/vbx_cnn_api.c
  VectorBlox-SDK/drivers/vectorblox/vbx_cnn_model.c

실제 mmap 장치는 vbx_cnn_api.c의 DMA_DEV가 결정한다.

  #define DMA_DEV "udmabuf-vbx64"

Makefile은 이 설정이 아니면 빌드를 중단한다.


2. WSL 빌드
-----------

  cd /mnt/d/VectorBlox_Icicle_Code/Polarfire_SoC_testkit/yolo_dma_c

  make clean
  make \
    CC=riscv64-linux-gnu-gcc \
    SDK_ROOT=/home/minseo/vectorblox-build/VectorBlox-SDK

  make inspect \
    SDK_ROOT=/home/minseo/vectorblox-build/VectorBlox-SDK

필요한 컴파일 매크로:

  -DVBX_SOC_DRIVER
  -DMSS_DDR
  -DUSE_INTERRUPTS=0

postprocess_yolov8n.c 때문에 libm을 링크한다. JPEG 입력을 사용하지 않으므로
libjpeg는 링크하지 않는다.


3. 보드 실행 전 overlay
-----------------------

재부팅 후:

  cd /root/VectorBlox-SDK/example/soc-c
  make overlay

  cd /root/VectorBlox-SDK/example/soc-c/dts
  bash overlay.sh add vbx_dma64.dtbo

  cat /sys/class/u-dma-buf/udmabuf-vbx64/size

예상값:

  67108864


4. 보드 실행
------------

  chmod +x /root/yolo-dma-hello

  /root/yolo-dma-hello \
    /root/yolo_dma_c/models/pruned_kd_single.vnnx \
    0.30 0.40

정상 실행 시 다음 항목이 출력된다.

  DATA_BYTES
  ALLOCATE_BYTES
  MODEL_SANITY = 0
  classes
  DMA input bytes
  inference-only
  output[0..2] shape/scale/zero/checksum
  detections
  Hello, YOLO DMA inference!
