# Changelog

All notable user-facing changes to `image-evaluator` are recorded here.

## [0.1.0a1] - 2026-09-10

### Added

- Explicit `--metrics` selection with lazy loading for six metrics: Aesthetic,
  CLIP similarity, ArcFace distance, LPIPS, SSIM, and PSNR.
- Strict filename-stem pairing for directory evaluation.
- Pairwise fidelity documentation and reproducible command examples.

### Changed

- Python support is now 3.11 through 3.14.
- ONNX Runtime defaults to the CPU package on macOS and Linux; Linux users can
  replace it with the GPU runtime.
- Aesthetic scoring uses the OpenAI weight configuration with QuickGELU.

### Fixed

- Pairwise metrics reject mismatched dimensions instead of resizing silently.
- SSIM rejects images smaller than its 11 × 11 window instead of returning
  `NaN` with a successful exit code.

### Known limitations

- Windows has one external Python 3.12 smoke test covering CLIP, LPIPS, SSIM,
  and PSNR; complete Windows support remains unverified.
- The default CLIP model is intended for English prompts; multilingual prompt
  quality has not been validated.
- FID and KID are planned for M4 and are not part of this preview.
