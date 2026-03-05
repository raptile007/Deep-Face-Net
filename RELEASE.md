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
