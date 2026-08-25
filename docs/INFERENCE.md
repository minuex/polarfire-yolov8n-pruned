# PolarFire board inference

After each reboot, apply the VectorBlox and 64 MiB DMA overlays:

```sh
cd /root/VectorBlox-SDK/example/soc-c
make overlay
cd dts
bash overlay.sh add vbx_dma64.dtbo
```

Confirm the DMA pool:

```sh
cat /sys/class/u-dma-buf/udmabuf-vbx64/size
```

Expected value:

```text
67108864
```

Run either implementation with the same interface:

```sh
./run-yolov8n-c MODEL.vnnx IMAGE.jpg 0.30 0.40
```

or:

```sh
./run-yolov8n-cpp MODEL.vnnx IMAGE.jpg 0.30 0.40
```

Arguments are model path, JPEG path, confidence threshold, and NMS IoU
threshold. Models and images are intentionally excluded from this repository.

The model is expected to have one UINT8 input and three INT8 YOLOv8 outputs.
The output channel count is interpreted as 64 DFL channels plus the dynamic
number of classes.
