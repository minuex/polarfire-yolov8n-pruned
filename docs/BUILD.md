# Build

## Requirements

- RISC-V GNU cross compiler (`riscv64-linux-gnu-gcc/g++`)
- VectorBlox SDK matching the VNNX graph version used by the board
- target RISC-V libjpeg headers and `libjpeg.so.9`
- SDK driver configured for the 64 MiB DMA device

The default paths in both Makefiles are:

```text
SDK_ROOT=$HOME/vectorblox-build/VectorBlox-SDK
JPEG_SYSROOT=$HOME/icicle-target-sysroot
```

Override them on the command line when needed.

## Pure C

```sh
cd soc-c
make clean
make CC=riscv64-linux-gnu-gcc
make inspect
```

## C++

```sh
cd soc-cpp
make clean
make CC=riscv64-linux-gnu-gcc CXX=riscv64-linux-gnu-g++
make inspect
```

Both Makefiles compile the SDK `example/postprocess/image.c` implementation
and link against `libjpeg` and `libm`. The generated executables target
RISC-V 64-bit Linux with the lp64d ABI.

The Makefiles stop if `vbx_cnn_api.c` is not configured with:

```c
#define DMA_DEV "udmabuf-vbx64"
```

The corresponding source change is recorded in
`patches/vbx_cnn_api_dma64.patch`.
