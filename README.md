# PolarFire SoC YOLOv8 VNNX inference

YOLOv8 VNNX inference and postprocessing sources for the Microchip PolarFire
SoC Icicle Kit with a VectorBlox accelerator.

This public repository contains two equivalent runner implementations:

- `soc-cpp/`: C++ runner
- `soc-c/`: pure-C runner

Both use the same YOLOv8 DFL decode, confidence filtering, and NMS code in
`postprocess/`. The number of classes is inferred from the VNNX output channel
count, so single-class and multi-class models use the same source.

## Repository layout

```text
postprocess/                 shared YOLOv8 postprocessing
soc-cpp/                     C++ JPEG/VNNX runner and Makefile
soc-c/                       pure-C JPEG/VNNX runner and Makefile
patches/                     VectorBlox 64 MiB DMA device patch
docs/                        build and board execution instructions
```

Models, datasets, test images, compiled binaries, the VectorBlox SDK, and
system/Pose integration code are intentionally not included.

See [docs/BUILD.md](docs/BUILD.md) and
[docs/INFERENCE.md](docs/INFERENCE.md).
