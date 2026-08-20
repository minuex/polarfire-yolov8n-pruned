CC ?= riscv64-linux-gnu-gcc
SDK_ROOT ?= /home/minseo/vectorblox-build/VectorBlox-SDK

VBX_DIR := $(SDK_ROOT)/drivers/vectorblox
POST_DIR ?= .
TARGET := yolo-dma-hello
BUILD_DIR := obj

COMMON_FLAGS := -O3 -std=c11 -Wall -Wextra -I$(VBX_DIR) \
                -I$(POST_DIR) -DVBX_SOC_DRIVER -DMSS_DDR -DUSE_INTERRUPTS=0
APP_CFLAGS ?= $(COMMON_FLAGS) -Werror
SDK_CFLAGS ?= $(COMMON_FLAGS)

APP_OBJS := $(BUILD_DIR)/yolo_dma_hello.o \
            $(BUILD_DIR)/postprocess_yolov8n.o
SDK_OBJS := $(BUILD_DIR)/vbx_cnn_api.o $(BUILD_DIR)/vbx_cnn_model.o
OBJS := $(APP_OBJS) $(SDK_OBJS)

.PHONY: all clean inspect check-dma64

all: check-dma64 $(TARGET)

# 실제 mmap 장치는 환경변수가 아니라 SDK vbx_cnn_api.c의 DMA_DEV가 결정한다.
check-dma64:
	@grep -q 'DMA_DEV "udmabuf-vbx64"' $(VBX_DIR)/vbx_cnn_api.c || \
	  (echo 'ERROR: SDK vbx_cnn_api.c is not configured for udmabuf-vbx64'; exit 1)

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/yolo_dma_hello.o: yolo_dma_hello.c | $(BUILD_DIR)
	$(CC) $(APP_CFLAGS) -c $< -o $@

$(BUILD_DIR)/postprocess_yolov8n.o: $(POST_DIR)/postprocess_yolov8n.c \
                                   $(POST_DIR)/postprocess_yolov8n.h | $(BUILD_DIR)
	$(CC) $(APP_CFLAGS) -c $(POST_DIR)/postprocess_yolov8n.c -o $@

$(BUILD_DIR)/vbx_cnn_api.o: $(VBX_DIR)/vbx_cnn_api.c | $(BUILD_DIR)
	$(CC) $(SDK_CFLAGS) -c $< -o $@

$(BUILD_DIR)/vbx_cnn_model.o: $(VBX_DIR)/vbx_cnn_model.c | $(BUILD_DIR)
	$(CC) $(SDK_CFLAGS) -c $< -o $@

$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $@ -lm

inspect: $(TARGET)
	file $(TARGET)
	riscv64-linux-gnu-readelf -h $(TARGET) | grep -E 'Class:|Machine:|Flags:'
	riscv64-linux-gnu-readelf -d $(TARGET) | grep NEEDED

clean:
	rm -rf $(TARGET) $(BUILD_DIR)
