# Fréchet Inception Distance (FID)

## Facts

| Property | Value |
| :--- | :--- |
| **Category** | Generative Distribution Fidelity |
| **CLI Metric** | `fid` |
| **Inputs** | `--reference <dir>`, `--image <dir>` (Both arguments strictly require existing directories) |
| **Output** | Non-negative scalar float in `[0.0, +inf)` |
| **Direction** | **Lower is better** (0.0 represents identical feature distributions) |
| **Formula** | $d^2((\mu_r, \Sigma_r), (\mu_g, \Sigma_g)) = \|\mu_r - \mu_g\|_2^2 + \text{Tr}\left(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\right)$ |
| **Backbone** | Inception-v3 (TorchScript weights from StyleGAN2-ADA / clean-fid) |
| **Protocol** | `clean-fid==0.1.35`, `mode="clean"`, `model_name="inception_v3"`, device CPU by default |
| **Upstream Reference** | Parmar et al., "On Aliased Resizing and Surprising Subtleties in GAN Evaluation", CVPR 2022 |

---

## Scientific Background & Protocol Rationale

Fréchet Inception Distance (Heusel et al., NeurIPS 2017) measures the Wasserstein-2 distance between two multivariate Gaussian distributions fitted to the 2,048-dimensional activations of the penultimate pooling layer of an Inception-v3 network:
$$\text{FID} = \|\mu_r - \mu_g\|_2^2 + \text{Tr}\left(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\right)$$
where $(\mu_r, \Sigma_r)$ and $(\mu_g, \Sigma_g)$ denote the empirical mean vector and covariance matrix of reference and generated images, respectively.

### Why `clean-fid`?
Standard implementations of FID (such as `pytorch-fid` or TensorFlow official scripts) suffer from subtle implementation artifacts:
1. **Aliasing during resizing**: Downsampling high-resolution images to the Inception input resolution ($299 \times 299$) without proper anti-aliasing introduces high-frequency artifacts that pollute deep feature representations.
2. **Library-dependent resampling filters**: PIL, OpenCV, PyTorch, and TensorFlow use differing filter implementations, producing significant numerical shifts (up to 5–10 FID points) on the identical image dataset.
3. **JPEG compression artifacts**: Storing intermediate images as JPEG alters activation distributions.

`clean-fid` (Parmar et al., CVPR 2022) addresses these issues by standardizing bilinear downsampling with a strict anti-aliasing filter before feature extraction. To guarantee reproducibility, `image-evaluator` fixes:
- **Library Version**: `clean-fid==0.1.35` (enforced at runtime via metadata checks).
- **Mode**: `"clean"` (anti-aliased bilinear interpolation; distinct from `"legacy_pytorch"` or `"legacy_tensorflow"`).
- **Model**: `"inception_v3"` (2,048-dimensional feature extractor).
- **Device**: CPU by default, with explicit device injection supported in library code without silent fallback.

> [!IMPORTANT]
> FID scores computed with `mode="clean"` **cannot be directly compared** to FID scores reported from legacy `pytorch-fid` or TensorFlow pipelines. When reporting benchmark numbers, always cite the backend and protocol explicitly.

---

## Directory Role & Input Contract

Unlike pairwise metrics (such as LPIPS, SSIM, or PSNR), FID evaluates **population-level distributions** rather than image-by-image correspondences:

1. **Role Binding**:
   - `--reference`: Reference / ground-truth image directory ($f_{\text{dir1}}$, yielding $\mu_r, \Sigma_r$).
   - `--image`: Generated image directory ($f_{\text{dir2}}$, yielding $\mu_g, \Sigma_g$).
2. **Strict Directory Requirement**:
   Both `--image` and `--reference` must be existing directories. Passing a file path to either argument is rejected immediately during CLI parsing with a non-zero exit code:
   ```text
   error: --image must be a directory when 'fid' metric is selected, got file: 'sample.png'
   ```
3. **No Filename Pairing**:
   FID does not require filename stem matching. Files in `--image` and `--reference` can have completely different names, structures, and sample counts ($N_{\text{ref}} \ne N_{\text{gen}}$).
4. **Recursive Image Discovery**:
   Both folders are scanned recursively for supported image formats matching upstream `clean-fid`:
   `.bmp`, `.jpg`, `.jpeg`, `.pgm`, `.png`, `.ppm`, `.tif`, `.tiff`, `.webp`, `.npy`.
5. **Fail-Fast Image Validation**:
   Prior to feature extraction, every discovered file is opened and verified via PIL (`img.verify()` and `img.convert("RGB")`). Any corrupted or unreadable image causes the entire run to fail immediately with `ValueError`, preventing silent sample omissions.

---

## Sample Size Sensitivity & Audit Metadata

FID is a **biased estimator**: for finite sample sizes, empirical covariance estimates systematically overestimate the true population distance.

- **Minimum Mathematical Bound ($N \ge 2$)**:
  At least 2 valid images are required in each directory to compute sample covariance. Any directory with fewer than 2 images raises `ValueError`.
- **Sample Sensitivity Warning**:
  On every FID evaluation, `image-evaluator` emits a clear warning reminding users of finite-sample sensitivity:
  ```text
  UserWarning: FID is sensitive to sample size (Nref=3, Ngen=3). Statistical reliability requires task-specific convergence or repeated trials; Nref and Ngen are recorded for auditability.
  ```
- **Benchmark Sample Counts**:
  - Canonical benchmark publication standard: $N = 50,000$ (e.g. COCO-30k or ImageNet-50k). For smaller datasets, finite-sample bias increases significantly; pairing with unbiased KID is recommended.
- **Audit Metadata Output**:
  Every CLI evaluation prints protocol metadata alongside the scalar FID score to prevent incomparable evaluations:
  ```text
  FID: 12.345678 (backend=clean-fid, version=0.1.35, mode=clean, model=inception_v3, device=cpu, Nref=1000, Ngen=1000)
  ```

---

## Invocation Examples

### 1. Standalone FID Evaluation
```bash
image-evaluator \
    --metrics fid \
    --reference /path/to/real_images/ \
    --image /path/to/generated_images/
```

### 2. Multi-Metric Joint Evaluation (Distribution + Pairwise Fidelity)
When combining FID with pairwise metrics (`lpips`, `ssim`, `psnr`), pairwise metrics require matched filename stems across directories, while FID independently assesses overall distribution distances:
```bash
image-evaluator \
    --metrics lpips ssim psnr fid \
    --reference /path/to/reference_dir/ \
    --image /path/to/generated_dir/
```

### 3. Multi-Metric Joint Evaluation (Aesthetic + Text Alignment + FID)
```bash
image-evaluator \
    --metrics aesthetic clip fid \
    --prompt "a photorealistic portrait of an astronaut" \
    --reference /path/to/reference_dir/ \
    --image /path/to/generated_dir/
```

---

## Known Limitations & Boundaries

1. **Not a Per-Image Metric**: FID cannot score a single generated image. Evaluating image-to-image quality should be done via `lpips`, `ssim`, or `psnr`.
2. **Computational Overhead**: Inception-v3 feature extraction scales linearly with sample count. For large datasets ($N \ge 50,000$) on CPU, evaluation may take several minutes; running on GPU (`CUDA`) is recommended for large-scale benchmarks.
3. **Cross-Backend Invariance**: Never compare an FID score computed via `clean-fid` against numbers from `pytorch-fid`, `torch-fidelity`, or TensorFlow scripts without noting the backend discrepancy.
