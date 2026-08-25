CC ?= riscv64-linux-gnu-gcc
SDK_ROOT ?= $(HOME)/vectorblox-build/VectorBlox-SDK
JPEG_SYSROOT ?= $(HOME)/icicle-target-sysroot

VBX_DIR := $(SDK_ROOT)/drivers/vectorblox
IMAGE_SRC := $(SDK_ROOT)/example/postprocess/image.c
POST_DIR := ../postprocess
BUILD_DIR := obj
TARGET := run-yolov8n-c

COMMON_FLAGS := -O3 -std=c11 -Wall -Wextra \
	-I$(POST_DIR) -I$(VBX_DIR) -I$(JPEG_SYSROOT)/usr/include \
	-DVBX_SOC_DRIVER -DMSS_DDR -DUSE_INTERRUPTS=0
APP_CFLAGS ?= $(COMMON_FLAGS) -Werror
SDK_CFLAGS ?= $(COMMON_FLAGS)
LIBS := -L$(JPEG_SYSROOT)/usr/lib \
	-Wl,-rpath-link,$(JPEG_SYSROOT)/usr/lib -ljpeg -lm

OBJS := $(BUILD_DIR)/run_yolov8n.o \
	$(BUILD_DIR)/postprocess_yolov8n.o \
	$(BUILD_DIR)/image.o \
	$(BUILD_DIR)/vbx_cnn_api.o \
	$(BUILD_DIR)/vbx_cnn_model.o

.PHONY: all clean inspect check-dma64

all: check-dma64 $(TARGET)

check-dma64:
	@grep -q 'DMA_DEV "udmabuf-vbx64"' $(VBX_DIR)/vbx_cnn_api.c || \
	  (echo 'ERROR: SDK vbx_cnn_api.c is not configured for udmabuf-vbx64'; exit 1)

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/run_yolov8n.o: run_yolov8n.c $(POST_DIR)/postprocess_yolov8n.h | $(BUILD_DIR)
	$(CC) $(APP_CFLAGS) -c $< -o $@

$(BUILD_DIR)/postprocess_yolov8n.o: $(POST_DIR)/postprocess_yolov8n.c \
	$(POST_DIR)/postprocess_yolov8n.h | $(BUILD_DIR)
	$(CC) $(APP_CFLAGS) -c $< -o $@

$(BUILD_DIR)/image.o: $(IMAGE_SRC) | $(BUILD_DIR)
	$(CC) $(SDK_CFLAGS) -c $< -o $@

$(BUILD_DIR)/vbx_cnn_api.o: $(VBX_DIR)/vbx_cnn_api.c | $(BUILD_DIR)
	$(CC) $(SDK_CFLAGS) -c $< -o $@

$(BUILD_DIR)/vbx_cnn_model.o: $(VBX_DIR)/vbx_cnn_model.c | $(BUILD_DIR)
	$(CC) $(SDK_CFLAGS) -c $< -o $@

$(TARGET): $(OBJS)
	$(CC) $^ -o $@ $(LIBS)

inspect: $(TARGET)
	file $(TARGET)
	riscv64-linux-gnu-readelf -d $(TARGET) | grep NEEDED

clean:
	rm -rf $(TARGET) $(BUILD_DIR)
