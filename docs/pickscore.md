# PickScore (Human Preference Score)

## Facts

| Property | Value |
| :--- | :--- |
| **Category** | Human Preference & Text-to-Image Alignment |
| **CLI Metric** | `pickscore` |
| **Inputs** | `--image <file_or_dir>`, `--prompt "<text_prompt>"` (Required) |
| **Output** | Real scalar float (typical range `[15.0, 25.0]`) |
| **Direction** | **Higher is better** (higher values reflect greater human preference) |
| **Backbone** | CLIP ViT-H-14 (Vision Transformer Huge, ~986M parameters) |
| **Processor** | `laion/CLIP-ViT-H-14-laion2B-s32B-b79K` |
| **Model Weights** | `yuvalkirstain/PickScore_v1` (Hugging Face Hub) |
| **Underlying Stack** | Native `transformers` (`AutoModel`, `AutoProcessor`) + `torch` |
| **License** | MIT License |
| **Upstream Reference** | Kirstain et al., "Pick-a-Pic: An Open Dataset of User Preferences for Text-to-Image Generation", NeurIPS 2023 |

---

## Scientific Background & Scoring Formulation

Standard image-text similarity metrics (such as vanilla CLIP cosine similarity) measure broad semantic alignment learned via web-scraped contrastive pre-training. However, they frequently fail to capture nuanced human preferences regarding visual artifacts, fine-grained details, aesthetic balance, and prompt fidelity.

**PickScore** (Kirstain et al., NeurIPS 2023) directly addresses this gap. The authors collected **Pick-a-Pic**, a dataset of over 500,000 real-world human preferences on AI-generated images, and fine-tuned a CLIP ViT-H-14 backbone using a Bradley-Terry preference model objective.

### Mathematical Definition

Given an input text prompt $p$ and a generated image $x$:

1. **Feature Extraction**:
   $$e(p) = \text{TextEncoder}(p), \quad e(x) = \text{ImageEncoder}(x)$$

2. **$L_2$ Normalization**:
   $$\hat{e}(p) = \frac{e(p)}{\|e(p)\|_2}, \quad \hat{e}(x) = \frac{e(x)}{\|e(x)\|_2}$$

3. **Calibrated Dot-Product Scoring**:
   $$\text{PickScore}(p, x) = \exp(\text{logit\_scale}) \cdot (\hat{e}(p) \cdot \hat{e}(x))$$

Here, $\text{logit\_scale}$ is the learned temperature parameter optimized during preference training.

### Probabilistic Interpretation

When ranking $K$ candidate images $[x_1, \dots, x_K]$ generated from the same prompt $p$, the probability that image $x_i$ is preferred over image $x_j$ follows the logistic sigmoid of their score difference:
$$P(x_i \succ x_j) = \frac{1}{1 + \exp\left(-\left(\text{PickScore}(p, x_i) - \text{PickScore}(p, x_j)\right)\right)}$$

For multi-candidate selection, applying the softmax function over candidate scores yields the human preference distribution:
$$P(x_i) = \frac{\exp(\text{PickScore}(p, x_i))}{\sum_{k=1}^K \exp(\text{PickScore}(p, x_k))}$$

In single-image or directory evaluation mode, `image-evaluator` outputs the unnormalized calibrated scalar score $\text{PickScore}(p, x)$ directly.

---

## Input Contract & Execution Modes

PickScore requires both an image input and a descriptive text prompt:

1. **Single-Image Mode**:
   - Evaluates a single image against the specified `--prompt`.
   - Returns a single scalar score:
   ```bash
   image-evaluator --metrics pickscore --image sample.png --prompt "a photo of a cat in space"
   # Output:
   # PickScore: 21.4321
   ```

2. **Directory Batch Mode**:
   - Discovers all supported images (`.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`) in the target directory.
   - Evaluates each image independently against the common `--prompt`.
   - Outputs the arithmetic mean score across all valid images:
   ```bash
   image-evaluator --metrics pickscore --image ./generated/ --prompt "a photo of a cat in space"
   # Output:
   # PickScore: 20.8765
   ```

3. **Prompt Requirement & Validation (Fail-Fast)**:
   - `--prompt` is **mandatory** whenever `pickscore` is selected.
   - Omitting `--prompt` or providing an empty/whitespace string triggers immediate error termination (exit code 2):
     ```text
     error: --prompt is required when 'pickscore' metric is selected.
     ```
   - Passing `--prompt` when neither `clip` nor `pickscore` is selected is disallowed to prevent silent parameter misuse.

4. **Reference Independence**:
   - PickScore evaluates text-to-image preference; it does not consume `--reference`.
   - If `--reference` is passed alongside reference-based metrics (such as LPIPS, SSIM, FID, or KID), PickScore executes independently without conflict.

---

## Hardware Footprint & Offline Caching

| Resource | Specification |
| :--- | :--- |
| **Model Weights** | `model.safetensors` (~3.94 GB) |
| **Disk Storage** | Standard Hugging Face cache at `~/.cache/huggingface/hub` (configurable via `HF_HOME`) |
| **GPU VRAM (FP16)** | ~2.5 GB – 3.5 GB peak |
| **CPU RAM (FP32)** | ~4.5 GB – 5.0 GB peak |
| **Supported Devices** | CUDA (NVIDIA), MPS (Apple Silicon), CPU |

### Lazy Loading & Zero Overhead

`image-evaluator` enforces strict deferred loading:
- **No Background Imports**: Neither `transformers.AutoModel` nor model weights are loaded at startup.
- **Selective Activation**: If you execute other metrics (e.g. `--metrics lpips ssim psnr fid kid`), PickScore code and weights remain completely inactive.
- **Offline Reliability**: Once cached locally, subsequent executions are 100% offline with zero network calls.

---

## Multi-Metric Integration Examples

### 1. Multi-Modal Alignment & Aesthetic Triad
Evaluate visual beauty, semantic fidelity, and human preference in a single run:
```bash
image-evaluator --metrics aesthetic clip pickscore \
    --image output.png \
    --prompt "an oil painting of a cottage by a serene river"
```

### 2. Comprehensive Generation Assessment
Combine reference-based structural fidelity, distribution quality, and human preference:
```bash
image-evaluator --metrics lpips ssim fid kid pickscore \
    --reference ./real_images/ \
    --image ./generated_images/ \
    --prompt "a high-resolution photographic portrait"
```

---

## Scientific Caveats & Incomparability Boundaries

1. **Cross-Prompt Incomparability**:
   PickScore absolute values depend heavily on prompt length, vocabulary frequency, and text embedding magnitude. Comparing the absolute PickScore of an image conditioned on `"a cat"` against an image conditioned on `"a complex medieval steampunk clocktower"` is scientifically invalid. Always evaluate competing models or seeds using the **identical prompt**.
2. **Cross-Metric Incomparability**:
   PickScore values cannot be directly converted to or compared with scores from other reward models (e.g. ImageReward or HPS v2), as each model uses different pre-training backbones, calibration scales, and preference loss formulations.
3. **Distribution vs Individual Metrics**:
   PickScore evaluates individual text-image pairs. For evaluating overall dataset distribution fidelity and diversity, use PickScore in conjunction with dataset-level metrics (**FID** and **KID**).
