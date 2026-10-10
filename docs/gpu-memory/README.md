# GibbsDDRM GPU memory analysis

This folder records the analyzed experiments from
`GibbsDDRM_GPUv0.2_2026-10-10.docx` and the convolution selection behavior
verified against the relevant PyTorch source.

- [Measured GPU memory changes](memory-results.md)
- [cuDNN convolution selection with benchmark disabled](cudnn-convolution.md)

The measurements are from the reported Python experiments. The C++ CPU
comparison results are recorded in [../MIGRATION.md](../MIGRATION.md).
