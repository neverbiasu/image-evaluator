# Kernel Inception Distance (KID)

## Facts

| Property | Value |
| :--- | :--- |
| **Category** | Generative Distribution Fidelity |
| **CLI Metric** | `kid` |
| **Inputs** | `--reference <dir>`, `--image <dir>` (Both arguments strictly require existing directories) |
| **Output** | Finite scalar float in `(-inf, +inf)` (unbiased estimator admits negative values near zero) |
| **Direction** | **Lower is better** (0.0 indicates identical feature distributions) |
| **Kernel Function** | Polynomial kernel $k(x, y) = \left(\frac{1}{d} x^T y + 1\right)^3$ where $d = 2,048$ |
| **Estimator** | Minimum-variance unbiased U-statistic of squared Maximum Mean Discrepancy ($\text{MMD}_u^2$) |
| **Backbone** | Inception-v3 (TorchScript weights from StyleGAN2-ADA / clean-fid) |
| **Protocol** | `clean-fid==0.1.35`, `mode="clean"`, `model_name="inception_v3"`, `num_subsets=100`, `max_subset_size=1000`, `seed=0`, device CPU by default |
| **Upstream References** | Bińkowski et al., "Demystifying MMD GANs", ICLR 2018; Parmar et al., "On Aliased Resizing and Surprising Subtleties in GAN Evaluation", CVPR 2022 |

---

## Scientific Background & Protocol Rationale

Kernel Inception Distance (Bińkowski et al., ICLR 2018) measures the discrepancy between reference and generated image distributions by computing the squared Maximum Mean Discrepancy ($\text{MMD}^2$) between Inception-v3 feature representations ($d = 2,048$) under a cubic polynomial kernel:
$$k(x, y) = \left(\frac{1}{d} x^T y + 1\right)^3$$

Given reference features $X = \{x_1, \dots, x_m\}$ and generated features $Y = \{y_1, \dots, y_m\}$, the unbiased U-statistic estimator is defined as:
$$\text{MMD}_u^2(X, Y) = \frac{1}{m(m-1)} \sum_{i=1}^m \sum_{j \ne i}^m k(x_i, x_j) + \frac{1}{m(m-1)} \sum_{i=1}^m \sum_{j \ne i}^m k(y_i, y_j) - \frac{2}{m^2} \sum_{i=1}^m \sum_{j=1}^m k(x_i, y_j)$$

### Why KID vs FID?

While FID assumes that deep feature representations follow a single multivariate Gaussian distribution $\mathcal{N}(\mu, \Sigma)$, KID has two fundamental statistical advantages:
1. **Non-parametric Kernel MMD**: KID does not assume Gaussianity. The polynomial kernel implicitly captures higher-order moments of the distribution, making it sensitive to multimodality and fine-grained distribution shifts.
2. **Unbiased Finite-Sample Estimation**: FID has significant finite-sample bias: $\mathbb{E}[\widehat{\text{FID}}] > \text{FID}^*$ when $N < 50,000$. In contrast, the U-statistic estimator $\text{MMD}_u^2$ is **unbiased for any sample size** $m \ge 2$:
   $$\mathbb{E}\left[\text{MMD}_u^2(X, Y)\right] = \text{MMD}^2(P_r, P_g)$$
   While KID eliminates finite-sample bias for any $m \ge 2$, note that sampling variance remains non-negligible at smaller sample counts.

### Why `clean-fid`?

Just as with FID, standard KID implementations in older packages suffer from aliasing during downsampling and implementation differences across image decoding libraries. `image-evaluator` fixes the pipeline to `clean-fid==0.1.35` under `mode="clean"`, ensuring anti-aliased bilinear resampling before Inception-v3 extraction.

---

## Deterministic Seed Contract (`seed=0`)

KID computes the mean of $\text{MMD}_u^2$ across $M = 100$ randomly drawn subsets of size $m = \min(N_{\text{ref}}, N_{\text{gen}}, 1000)$.

In native libraries, calling KID repeatedly without a fixed seed generates different floating-point values due to random subset sampling. To guarantee **scientific reproducibility**:
- **Default Seed**: `image-evaluator` defaults strictly to `seed=0`.
- **Bit-Exact Reproducibility**: Running the identical command on identical directories produces bit-exact, identical floating-point scores down to the last decimal digit across multiple runs.
- **Random State Isolation**: `KIDPredictor` captures global NumPy random state via `np.random.get_state()` before seeding and restores it in a `finally` block via `np.random.set_state()`, preventing global random state leakage into user scripts or downstream tasks.

---

## Unbiased Estimators and Finite Negative Values

Because $\text{MMD}_u^2$ is an unbiased estimator of a non-negative quantity $\text{MMD}^2(P_r, P_g) \ge 0$, sample estimates fluctuate around the true population value with variance $\sigma^2 > 0$:
- When reference and generated distributions are identical ($P_r = P_g$), the true population distance is $\text{MMD}^2 = 0$.
- By symmetry of unbiased estimation, finite sample estimates will be slightly negative approximately 50% of the time (e.g. $-0.0008$).
- Similarly, on very small sample counts ($N = 3$), finite sample variance can produce negative estimates (e.g. $-0.096$).

> [!IMPORTANT]
> **Preserving Mathematical Integrity**: Artificial truncation of negative estimates to `0.0` destroys the unbiasedness of the estimator and introduces systematic positive bias. `image-evaluator` strictly preserves the true mathematical float output while validating that the result is finite (`np.isfinite`), rejecting `NaN` or `Inf`.

---

## Directory Role & Input Contract

Like FID, KID evaluates population-level distributions:

1. **Role Binding**:
   - `--reference`: Reference / ground-truth image directory ($f_{\text{dir1}}$, sample size $N_{\text{ref}}$).
   - `--image`: Generated image directory ($f_{\text{dir2}}$, sample size $N_{\text{gen}}$).
2. **Strict Directory Requirement**:
   Both `--image` and `--reference` must be existing directories. Passing a file path to either argument raises a parser error immediately:
   ```text
   error: --image must be a directory when 'kid' metric is selected, got file: 'sample.png'
   ```
3. **No Filename Pairing**:
   KID does not require filename stem matching. Folder contents may have different filenames, structures, and sample counts ($N_{\text{ref}} \ne N_{\text{gen}}$).
4. **Recursive Image Discovery**:
   Recursively discovers all 10 upstream extensions: `.bmp`, `.jpg`, `.jpeg`, `.pgm`, `.png`, `.ppm`, `.tif`, `.tiff`, `.webp`, `.npy`.
5. **Fail-Fast Image Validation**:
   Every discovered file is opened and verified via PIL (`img.verify()` and `img.convert("RGB")`) or NumPy (`np.load`). Corrupted or unreadable files raise `ValueError` immediately.
6. **Sample Size Threshold ($N \ge 2$)**:
   Computing the U-statistic requires $m \ge 2$ samples (due to the $m-1$ denominator). If either folder contains fewer than 2 valid images, execution raises `ValueError`.

---

## Audit Metadata Output

Every CLI invocation prints protocol and execution parameters to ensure transparency:
```text
KID: -0.09649164229631424 (backend=clean-fid, version=0.1.35, mode=clean, model=inception_v3, device=cpu, num_subsets=100, max_subset_size=1000, seed=0, Nref=3, Ngen=3)
```

Audit fields recorded:
- `backend`: Underlying implementation library (`clean-fid`).
- `version`: Exact installed package version (`0.1.35`).
- `mode`: Resampling and preprocessing pipeline (`clean`).
- `model`: Feature extraction backbone (`inception_v3`).
- `device`: Hardware acceleration device (`cpu`).
- `num_subsets`: Number of Monte Carlo subset iterations ($M = 100$).
- `max_subset_size`: Maximum subset size cap ($m \le 1000$).
- `seed`: Random seed for subset sampling (`0`).
- `Nref`: Valid sample count in reference directory.
- `Ngen`: Valid sample count in generated directory.

---

## Invocation Examples

### 1. Standalone KID Evaluation
```bash
image-evaluator \
    --metrics kid \
    --reference /path/to/real_images/ \
    --image /path/to/generated_images/
```

### 2. Dual Distribution Evaluation (FID + KID)
Evaluate both Gaussian Wasserstein-2 distance (FID) and non-parametric polynomial MMD (KID):
```bash
image-evaluator \
    --metrics fid kid \
    --reference /path/to/reference_dir/ \
    --image /path/to/generated_dir/
```

### 3. Comprehensive Evaluation (Distribution + Pairwise Fidelity)
```bash
image-evaluator \
    --metrics lpips ssim psnr fid kid \
    --reference /path/to/reference_dir/ \
    --image /path/to/generated_dir/
```

---

## Known Limitations & Boundaries

1. **Not a Per-Image Metric**: KID evaluates population-level distribution distance; it cannot evaluate image-to-image perceptual similarity.
2. **Subset Computational Complexity**: Each of the 100 subsets performs an $O(m^2)$ kernel matrix computation where $m = \min(N, 1000)$. While fast for typical dataset sizes, large subsets on CPU may take several seconds.
3. **Cross-Library Discrepancies**: KID values depend on kernel degree, scaling factor ($1/d$), and anti-aliased resizing. Do not directly compare numbers against implementations using different resizing or unstandardized Inception weights.
