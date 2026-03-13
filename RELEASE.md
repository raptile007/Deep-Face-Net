# Deep Face Net v2.2.1

## What's New

### Advanced Face Enhancement Migration (ONNX)
- Migrated GFPGAN face enhancer backend to use pure `onnxruntime` instead of the `gfpgan` python package.
- Removed complex source build requirements for `BasicSR` and `GFPGAN`.
- Dramatically simplified installation process for users wanting enhancement features.
- Switched target enhancement model to the updated `GFPGANv1.4.onnx` (~336MB) available on HuggingFace.

---

# Deep Face Net v2.2.0
## What's New

### Advanced Face Enhancement (GFPGAN)
- Integrated **GFPGANv1.4** for high-quality face restoration and sharpening
- Three new processing modes available in offline processing:
  - **Swap Only**: Fast face swapping (default)
  - **Enhance Only**: Sharpen existing faces without swapping
  - **Swap + Enhance**: Swap faces and immediately enhance the result
- GFPGAN model automatically tracked in the Model Manager
- Dynamic UI updates based on GFPGAN model availability

## Requirements

No new dependencies. All existing requirements remain the same.
