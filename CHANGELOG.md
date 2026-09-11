# Changelog

All notable user-facing changes to `image-evaluator` are recorded here.

## [0.2.0] - 2026-09-11

### Added

- Dataset distribution metrics: Fréchet Inception Distance (FID) and Kernel Inception
  Distance (KID) using clean-fid 0.1.35 under Inception-v3 clean mode with deterministic seed=0.
- Human preference alignment metric: PickScore using official `yuvalkirstain/PickScore_v1`
  weights with scaled logit scoring for text-to-image quality assessment.
- Joint multi-metric evaluation across all 9 quality dimensions with lazy loading and
  bit-exact score conservation.
- Authoritative documentation manuals for FID (`docs/fid.md`), KID (`docs/kid.md`), and
  PickScore (`docs/pickscore.md`).

### Changed

- Expanded `--metrics` CLI options to accept `fid`, `kid`, and `pickscore`.
- Enforced Fail-Fast input validation: `--prompt` is required when `pickscore` is selected;
  `--image` and `--reference` must be existing directories when `fid` or `kid` is selected.
- Updated README.md with comprehensive 9-metric workflow diagram, selection guide, and
  installation commands for the 0.2.0 release.

### Fixed

- Eliminated sample-order sensitivity and non-deterministic random state leakage in KID evaluation.
- Resolved macOS spawn pickle issue by configuring worker process safety.

### Known limitations

- Windows has an external Python 3.12 smoke test covering CLIP, LPIPS, SSIM, and PSNR;
  complete Windows support remains unverified.
- The default CLIP and PickScore models are trained primarily on English text prompts;
  multilingual prompt quality has not been independently benchmarked.

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
