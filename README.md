# image-evaluator

`image-evaluator` is a lightweight CLI utility for generative AI practitioners and researchers. It evaluates fifteen core quality dimensions of synthetic images: predicted visual aesthetics, text-to-image semantic alignment, subject and pairwise visual semantic fidelity (CLIP-I), self-supervised structural similarity (DINOv2), image editing directional change (Directional CLIP), facial identity preservation, pairwise image fidelity (deep perceptual distance LPIPS, structural similarity SSIM, and peak signal-to-noise ratio PSNR), dataset distribution fidelity (Fréchet Inception Distance FID and Kernel Inception Distance KID), visual question answering alignment (VQAScore), and human preference alignment (PickScore, HPS v2.1, and ImageReward).

## Evaluation Workflow

```mermaid
flowchart LR
    A[Task Goal] --> B{Choose --metrics}
    B -->|aesthetic| C[LAION Aesthetic]
    B -->|clip| D[CLIP Similarity]
    B -->|clip_i| W[CLIP-I Similarity]
    B -->|dino_similarity| Y[DINOv2 Similarity]
    B -->|directional_clip| U[Directional CLIP]
    B -->|arcface| E[ArcFace Distance]
    B -->|lpips| F[LPIPS Distance]
    B -->|ssim| G[SSIM Similarity]
    B -->|psnr| H[PSNR Ratio]
    B -->|fid| O[FID Distance]
    B -->|kid| P[KID Distance]
    B -->|pickscore| S[PickScore Preference]
    B -->|hpsv2| HPS[HPS v2.1 Preference]
    B -->|image_reward| IR[ImageReward Preference]
    B -->|vqascore| VQA[VQAScore Posterior]
    C --> I[Float, Higher Better]
    D --> J[Cosine Sim, Higher Better]
    W --> X[Cosine Sim, Higher Better]
    Y --> Z[Cosine Sim, Higher Better]
    U --> V[Directional Sim, Higher Better]
    E --> K[Cosine Dist, Lower Better]
    F --> L[Distance, Lower Better]
    G --> M[Index, Higher Better]
    H --> N[dB, Higher Better]
    O --> Q[Wasserstein-2, Lower Better]
    P --> R[MMD, Lower Better]
    S --> T[Calibrated Likelihood, Higher Better]
    HPS --> HPS_OUT[Cosine Sim, Higher Better]
    IR --> IR_OUT[Continuous Reward, Higher Better]
    VQA --> VQA_OUT[Calibrated Posterior, Higher Better]
```

## Metric Selection

The toolkit registers 15 core metrics backed by an immutable `MetricRegistry`. Each metric provides formal input contracts, task and objective discovery tags, and execution protocols:

| Metric ID | Display Name | Tasks | Objectives | Required Inputs | Direction | Technical Details |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `aesthetic` | LAION Aesthetic Score | `text_to_image`, `image_editing` | `aesthetic_quality` | `image` | Higher is better | [docs/aesthetic-score.md](docs/aesthetic-score.md) |
| `clip` | CLIP Score | `text_to_image`, `image_editing` | `text_image_alignment` | `image`, `prompt` | Higher is better | [docs/clip-similarity.md](docs/clip-similarity.md) |
| `clip_i` | CLIP Image-Image Similarity | `image_editing`, `subject_driven_generation` | `fidelity`, `identity_preservation` | `image`, `reference_image` | Higher is better | [docs/clip-i-similarity.md](docs/clip-i-similarity.md) |
| `dino_similarity` | DINOv2 Image-Image Similarity | `image_editing`, `subject_driven_generation` | `fidelity`, `structural_similarity` | `image`, `reference_image` | Higher is better | [docs/dino-similarity.md](docs/dino-similarity.md) |
| `directional_clip` | Directional CLIP | `image_editing` | `edit_direction_alignment` | `image`, `reference_image`, `prompt`, `source_prompt` | Higher is better | [docs/directional-clip.md](docs/directional-clip.md) |
| `arcface` | ArcFace Distance | `face_generation`, `face_editing` | `identity_preservation` | `image`, `reference_image` | Lower is better | [docs/arcface-distance.md](docs/arcface-distance.md) |
| `lpips` | Learned Perceptual Patch Similarity | `image_editing`, `image_reconstruction`, `super_resolution` | `perceptual_similarity` | `image`, `reference_image` | Lower is better | [docs/pairwise-fidelity.md](docs/pairwise-fidelity.md) |
| `ssim` | Structural Similarity Index Measure | `image_editing`, `image_reconstruction`, `super_resolution` | `structural_similarity` | `image`, `reference_image` | Higher is better | [docs/pairwise-fidelity.md](docs/pairwise-fidelity.md) |
| `psnr` | Peak Signal-to-Noise Ratio | `image_editing`, `image_reconstruction`, `super_resolution` | `pixel_fidelity` | `image`, `reference_image` | Higher is better | [docs/pairwise-fidelity.md](docs/pairwise-fidelity.md) |
| `fid` | Fréchet Inception Distance | `text_to_image`, `unconditional_generation` | `distribution_similarity` | `image_collection`, `reference_collection` | Lower is better | [docs/fid.md](docs/fid.md) |
| `kid` | Kernel Inception Distance | `text_to_image`, `unconditional_generation` | `distribution_similarity` | `image_collection`, `reference_collection` | Lower is better | [docs/kid.md](docs/kid.md) |
| `pickscore` | PickScore | `text_to_image` | `human_preference` | `image`, `prompt` | Higher is better | [docs/pickscore.md](docs/pickscore.md) |
| `hpsv2` | HPS v2.1 | `text_to_image`, `subject_driven_generation` | `human_preference` | `image`, `prompt` | Higher is better | [docs/hpsv2.md](docs/hpsv2.md) |
| `image_reward` | ImageReward | `text_to_image`, `subject_driven_generation` | `human_preference` | `image`, `prompt` | Higher is better | [docs/image-reward.md](docs/image-reward.md) |
| `vqascore` | VQAScore | `text_to_image`, `subject_driven_generation` | `alignment`, `compositionality` | `image`, `prompt` | Higher is better | [docs/vqascore.md](docs/vqascore.md) |

## Quick Start

### Installation

`image-evaluator` requires Python 3.11–3.14. Install the release via pip:

```bash
python -m pip install image-evaluator==0.4.0
```

The supported runtime path is macOS or Linux with CPU ONNX Runtime. Linux
users who want the GPU runtime can replace `onnxruntime` with
`onnxruntime-gpu` after installation. Windows is currently unverified.

#### Optional Modern Dependency Groups

For modern VQA and preference metrics, install the optional dependency groups:

```bash
# Optional dependencies for VQAScore (T5 architecture and accelerated inference)
python -m pip install 'image-evaluator[vqa]'

# Optional dependencies for modern preference and vision backends (e.g. timm)
python -m pip install 'image-evaluator[preference]'

# Union of all modern evaluation dependencies
python -m pip install 'image-evaluator[modern]'
```

### CLI Metric Discovery & Inspection

`image-evaluator` provides instant metric discovery and specification inspection with zero deep-model loading overhead (<20ms response):

1. **List all registered metrics**:
   ```bash
   image-evaluator list
   ```

2. **Filter metrics by task or objective**:
   ```bash
   image-evaluator list --task image_editing
   image-evaluator list --objective pixel_fidelity
   image-evaluator list -t image_reconstruction -o structural_similarity
   ```

3. **Inspect detailed metric specification**:
   ```bash
   image-evaluator show directional_clip
   ```

4. **Structured JSON output for pipeline discovery**:
   ```bash
   image-evaluator list --format json
   image-evaluator show directional_clip --format json
   ```

### Tutorial

The CLI enforces explicit metric selection via `--metrics` and initializes only selected models:
- `--metrics` (required): One or more of `aesthetic`, `clip`, `clip_i`, `dino_similarity`, `directional_clip`, `arcface`, `lpips`, `ssim`, `psnr`, `fid`, `kid`, `pickscore`, `hpsv2`.
- `--image` (required): Path to an image file or directory (must be a directory when `fid` or `kid` is selected).
- `--prompt`: Required when `clip`, `pickscore`, `hpsv2`, or `directional_clip` is selected; prohibited otherwise.
- `--prompt-src`: Required when `directional_clip` is selected; describes the source image before editing.
- `--reference`: Required when reference-based metrics (`directional_clip`, `clip_i`, `dino_similarity`, `arcface`, `lpips`, `ssim`, `psnr`, `fid`, `kid`) are selected; prohibited otherwise (must be a directory when `fid` or `kid` is selected).
- `--allow-download`: Opt-in flag to authorize automatic downloading of model weights from external sources if not locally cached (default: False). When omitted, evaluating an uncached model fails fast with a `DownloadNotAllowedError` to prevent unexpected network transfers and disk consumption.
- `--format`: Output format, either `text` (default) or `json` (for CI/CD and `jq` pipelines).

1. **Aesthetic evaluation only**:
   ```bash
   image-evaluator --metrics aesthetic --image path/to/image.png
   ```

2. **CLIP text alignment only**:
   ```bash
   image-evaluator --metrics clip --image path/to/image.png --prompt "a cat in oil painting style"
   ```

3. **Pairwise fidelity triad (LPIPS, SSIM, PSNR)**:
   ```bash
   image-evaluator --metrics lpips ssim psnr --image path/to/image.png --reference path/to/ref.png
   ```

4. **CLIP-I pairwise visual semantic fidelity**:
   ```bash
   image-evaluator --metrics clip_i --image path/to/generated.png --reference path/to/ref.png
   ```

5. **DINOv2 pairwise structural & visual semantic fidelity**:
   ```bash
   image-evaluator --metrics dino_similarity --image path/to/generated.png --reference path/to/ref.png
   ```

6. **Structured JSON output for CI/CD automation**:
   ```bash
   image-evaluator --metrics ssim psnr --image img.png --reference ref.png --format json | jq .metrics.ssim
   ```

7. **Dataset distribution metrics (FID and KID)**:
   ```bash
   image-evaluator --metrics fid kid \
       --reference path/to/real_images/ \
       --image path/to/generated_images/
   ```

8. **PickScore human preference evaluation**:
   ```bash
   image-evaluator --metrics pickscore \
       --image path/to/image.png \
       --prompt "a cat in oil painting style"
   ```

9. **HPS v2.1 human preference evaluation**:
   ```bash
   image-evaluator --metrics hpsv2 \
       --image path/to/image.png \
       --prompt "a cat in oil painting style"
   ```

10. **ImageReward human preference evaluation**:
   ```bash
   image-evaluator --metrics image_reward \
       --image path/to/image.png \
       --prompt "a cat in oil painting style"
   ```

11. **VQAScore alignment & compositionality evaluation**:
   ```bash
   image-evaluator --metrics vqascore \
       --image path/to/image.png \
       --prompt "a red sports car parked in front of a modern building"
   ```

12. **Multi-metric evaluation across folders**:
   ```bash
   image-evaluator --metrics aesthetic clip clip_i dino_similarity arcface lpips ssim psnr fid kid pickscore hpsv2 image_reward vqascore \
       --image path/to/images/ \
       --prompt "a cat in oil painting style" \
       --reference path/to/refs/
   ```

### Python SDK (Zero-Disk-I/O Memory Stream)

For single-image and pairwise metrics (`aesthetic`, `clip`, `clip_i`, `dino_similarity`, `directional_clip`, `arcface`, `lpips`, `ssim`, `psnr`, `pickscore`, `hpsv2`, `image_reward`, `vqascore`), evaluate in-memory `PIL.Image`, `torch.Tensor`, or `numpy.ndarray` objects directly without saving to disk (dataset metrics `fid` and `kid` require directory paths):

```python
import torch
from image_evaluator import evaluate, evaluate_detailed

t_img = torch.rand(3, 256, 256)
t_ref = torch.rand(3, 256, 256)

# 1. Standard Dictionary Return (100% backward compatible)
scores = evaluate(
    metrics=["ssim", "psnr"],
    image=t_img,
    reference=t_ref,
    device="cpu",
)
print(scores)  # {'ssim': 0.0012, 'psnr': 8.142}

# 2. Detailed Result API (structured result with timing & specs)
result = evaluate_detailed(
    metrics=["ssim", "psnr"],
    image=t_img,
    reference=t_ref,
    device="cpu",
)
print(result["ssim"])              # 0.0012 (Mapping protocol transparent access)
print(result.scores)               # {'ssim': 0.0012, 'psnr': 8.142}
print(result.duration_seconds)     # float: precise execution duration
print(result.to_json())            # RFC 8259 compliant JSON string

# 3. Model Weight Download Policy (Opt-In & Disclosure)
# Uncached model weights raise DownloadNotAllowedError by default.
# Pass allow_download=True to permit explicit downloads:
scores = evaluate(
    metrics=["ssim", "psnr"],
    image=t_img,
    reference=t_ref,
    allow_download=True,
)
```

#### Programmatic Metric Registry Discovery

Inspect registered metric specifications directly without importing heavy model backends:

```python
from image_evaluator.registry import filter_metrics, get_metric, list_metrics

# List all 10 registered metrics (<20ms response, zero heavy imports)
all_metrics = list_metrics()

# Filter metrics by task and objective
editing_metrics = filter_metrics(task="image_editing")
fidelity_metrics = filter_metrics(objective="pixel_fidelity")

# Inspect specification details
spec = get_metric("directional_clip")
print(spec.display_name)       # "Directional CLIP"
print(spec.score_direction)    # "higher_is_better"
print(spec.inputs.required)    # ("image", "reference_image", "prompt", "source_prompt")
```

### Batch Evaluation & Performance Best Practices

When evaluating multiple images, avoid running the CLI in shell loops (`for f in *.png; do ...`), which re-initializes deep models ($N$ times) and introduces massive cold-start friction (~25-35s for 10 images).

Instead, choose between two high-throughput native paradigms:

1. **Native CLI Directory Mode** (Single model initialization, ~2-3s for 10 images):
   ```bash
   image-evaluator --metrics clip --image ./generated/ --prompt "a golden retriever on a sunny lawn"
   image-evaluator --metrics lpips ssim psnr --image ./generated/ --reference ./ground_truth/
   ```

2. **Python Predictor Reuse** (Zero reloading overhead, memory-resident tensors/PIL, ~1.5-2.0s for 10 images):
   ```python
   from PIL import Image
   from image_evaluator import ClipScorePredictor, SSIMPredictor

   # Instantiate once; weights stay resident in memory/VRAM
   clip_pred = ClipScorePredictor(device="cpu")
   ssim_pred = SSIMPredictor()

   # High-throughput in-memory loop
   samples = [("img1.png", "prompt1"), ("img2.png", "prompt2")]
   for path, prompt in samples:
       score = clip_pred.evaluate_clip_score(path, prompt)
   ```

See the full [Task Selection Guide](docs-site/content/docs/guides/task-selection.mdx) and [Batch Performance Guide](docs-site/content/docs/guides/batch-performance.mdx) on the documentation site.

## Interpretation & Protocol Guidelines

1. **Protocol Consistency**: Always compare scores under identical model backbones and preprocessing pipelines.
2. **Relative Comparison**: Avoid universal absolute thresholds; interpret scores relative to a baseline control.
3. **Fail-Fast Spatial Dimension Policy**: Pairwise metrics (`lpips`, `ssim`, `psnr`) strictly reject mismatched image dimensions with `ValueError` to prevent artificial interpolation distortion. Align sizes beforehand via downsampling or super-resolution.
4. **SSIM Minimum Size**: SSIM requires both image dimensions to be at least 11 pixels because it uses the documented 11 × 11 Gaussian window. Smaller inputs fail with `ValueError`.
5. **Dataset Distribution Contract**: Both `fid` and `kid` require existing directories on both sides; passing single files is rejected immediately. Neither metric requires filename stem matching ($N_{\text{ref}} \ne N_{\text{gen}}$ is allowed).
6. **Sample Size Sensitivity & Unbiasedness**: FID is a biased estimator with finite-sample bias that increases significantly on smaller sample sizes (triggering a `UserWarning` reminding users of finite-sample sensitivity). KID is an unbiased U-statistic estimator that can produce small negative values near zero; these reflect normal statistical fluctuations and must not be truncated. KID defaults to deterministic `seed=0` for bit-exact reproducibility.
7. **Human Preference Alignment**: PickScore assesses text-image alignment against trained human preference choices from the Pick-a-Pic dataset (`yuvalkirstain/PickScore_v1`). Scores are uncalibrated logits where higher scores indicate stronger preference; win rate is evaluated strictly via the sigmoid difference between candidate pairs conditioned on identical prompts. PickScore strictly requires an explicit text prompt (`--prompt`).

## Release Status

| Area | `0.3.0` status |
| :--- | :--- |
| Metrics | Aesthetic, CLIP, ArcFace, LPIPS, SSIM, PSNR, FID, KID, PickScore |
| macOS | Verified on Apple Silicon with Python 3.11 |
| Linux | Clean install and test suite verified on GitHub Actions with Python 3.11 |
| Windows | External Python 3.12 smoke test passed for CLIP, LPIPS, SSIM, and PSNR; full support remains unverified |
| Dataset metrics | FID and KID fully implemented via clean-fid 0.1.35 under clean Inception-v3 protocol with deterministic seed=0 |
| Human preference | PickScore fully implemented via PickScore_v1 (yuvalkirstain/PickScore_v1) with calibrated likelihood logits |

See [CHANGELOG.md](CHANGELOG.md) for the accepted user-facing changes and
known limitations.
