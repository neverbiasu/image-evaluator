# Changelog

All notable user-facing changes to `image-evaluator` are recorded here.

## [0.4.0] - 2026-09-15

### Added

- Directional CLIP (ΔCLIP) metric: Added `DirectionalClipPredictor` and CLI option
  `--metrics directional_clip` with `--prompt-src` to evaluate relative semantic
  direction alignment in image editing tasks (StyleGAN-NADA / InstructPix2Pix).
- Numerical safeguards: Built-in degenerate delta vector protection ($\epsilon=10^{-6}$)
  returning `0.0` when image or text changes are near zero, preventing NaN values.
- Task selection and batch performance guides: Published interactive decision trees
  and benchmarked invocation patterns across documentation site and README.
- Executable documentation tests: Added `tests/test_docs_examples.py` ensuring all
  published SDK and CLI snippets execute deterministically without code drift.

### Changed

- Comprehensive documentation truth alignment: Removed all uncalibrated absolute
  thresholds across 18 docs-site manuals and core documentation, anchoring evaluation
  guidance in empirical task-specific baselines and relative comparisons.
- Version bump: Updated toolkit version to 0.4.0 across package configuration,
  runtime attributes, and documentation links.

### Fixed

- CLI expected input error handling: Gracefully intercepted 8 classes of user input
  errors (missing paths, corrupt images, dimension mismatches, missing required options)
  before model initialization, eliminating Traceback leakage on expected failure paths.
- Unix stream separation (Boss Decision 6A): Ensured stdout remains strictly 0 bytes
  on error under `--format json`, redirecting single-line error messages to stderr
  with non-zero exit codes.

## [0.3.0] - 2026-09-13


### Added

- Added a top-level Python `evaluate(...)` API with lazy predictor imports.
- Added in-memory PIL, PyTorch Tensor, and NumPy inputs for single-image and
  pairwise metrics; FID and KID remain directory-based metrics.
- Added CLI `--format json` output for automation and pipeline integration.
- Added the AI-ready documentation site with interactive metric examples and
  machine-readable documentation endpoints.

### Changed

- Propagated the public `device` argument consistently across predictors.
- Preserved floating-point Tensor precision and defined deterministic grayscale
  and RGBA channel conversion behavior.

### Fixed

- Enforced RFC 8259 JSON output by converting non-finite Python and NumPy values
  to `null` and serializing with `allow_nan=False`.
- Preserved Python and NumPy boolean values as JSON `true` and `false`.

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
