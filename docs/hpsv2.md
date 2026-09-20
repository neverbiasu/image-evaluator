# Human Preference Score v2.1 (HPS v2.1)

## Facts

| Property | Specification |
| :--- | :--- |
| **Category** | Human Preference & Text-Image Alignment |
| **CLI Metric** | `hpsv2` |
| **Inputs** | `--image` (file or directory), `--prompt` (string) |
| **Output** | Raw cosine similarity scalar in `[-1.0, 1.0]` (typical range `[0.15, 0.40]`) |
| **Direction** | Higher is better |
| **Model** | OpenCLIP ViT-H-14 fine-tuned checkpoint (`HPS_v2.1_compressed.pt`) |
| **Backend** | `open_clip` |
| **Weights Repository** | Hugging Face [`xswu/HPSv2`](https://huggingface.co/xswu/HPSv2) |
| **Install Extra** | `pip install 'image-evaluator[preference]'` |
| **Citation** | [Wu et al., NeurIPS 2023 (HPSv2)](https://arxiv.org/abs/2306.09341) |

## Meaning & Role

Human Preference Score v2.1 measures the alignment between a generated image and its textual prompt according to human aesthetic and preference choices. It builds upon OpenCLIP's `ViT-H-14` vision-language dual-tower architecture, fine-tuned on the Human Preference Dataset v2 (HPD v2) containing nearly 800,000 human preference comparisons across modern diffusion models:

$$S_{\text{HPS}}(I, T) = \frac{\mathbf{v}(I)}{\|\mathbf{v}(I)\|_2} \cdot \frac{\mathbf{t}(T)}{\|\mathbf{t}(T)\|_2}$$

where $\mathbf{v}(I) \in \mathbb{R}^{1024}$ and $\mathbf{t}(T) \in \mathbb{R}^{1024}$ represent the vision and text feature embeddings respectively extracted from the fine-tuned ViT-H-14 network.

## Non-Equivalence Notice: Not Interchangeable with PickScore or ImageReward

`hpsv2` is **not** numerically interchangeable with other preference or reward models:
- **PickScore**: Applies an MLP classification head with temperature scaling ($\approx 15 \sim 25$), outputting scores typically between $18.0$ and $24.0$.
- **ImageReward**: Uses a cross-attention multimodal transformer with an MLP regression head, outputting unbounded continuous real values (mean $\approx 0$, can be negative).
- **HPS v2.1**: Outputs raw cosine dot products (typically $0.20 \sim 0.35$).
- **HPS v2.0 vs v2.1**: This library freezes `HPS_v2.1_compressed.pt` as the sole official checkpoint to regularize against modern diffusion over-fitting.

## Architectural Detail & Preprocessing

- **Backbone**: OpenCLIP `ViT-H-14` (79.8M parameters vision, 1024 embedding dimension).
- **Preprocessing**: Resize and center-crop to $224 \times 224$, standard OpenCLIP ImageNet normalization.
- **Tokenizer**: OpenCLIP text tokenizer with context length 77.
- **Memory & Storage**: Checkpoint size is $\approx 1.84$ GiB (1.97 GB); CPU evaluation requires $\approx 3.5$ GB RAM.

## Invocation

### CLI Usage

Single image evaluation:
```bash
image-evaluator --metrics hpsv2 --image path/to/generated.png --prompt "a cute red panda wearing glasses"
```

Directory-level evaluation (arithmetic mean across all images):
```bash
image-evaluator --metrics hpsv2 --image path/to/folder/ --prompt "a photo of an astronaut on Mars"
```

Explicit opt-in for downloading uncached model weights:
```bash
image-evaluator --metrics hpsv2 --image generated.png --prompt "sunset over ocean" --allow-download
```

Structured JSON output:
```bash
image-evaluator --metrics hpsv2 --image generated.png --prompt "cyberpunk city" --format json
```

### Python SDK Usage

High-level evaluation:
```python
from image_evaluator import evaluate, evaluate_detailed

# 1. Basic evaluation
res = evaluate(
    metrics="hpsv2",
    image="path/to/generated.png",
    prompt="a digital painting of a mountain landscape",
)
print("HPS v2.1 Score:", res["hpsv2"])

# 2. Detailed result
result = evaluate_detailed(
    metrics="hpsv2",
    image="path/to/generated.png",
    prompt="a digital painting of a mountain landscape",
)
print(f"HPS v2.1: {result.scores['hpsv2']:.4f} (took {result.duration_seconds:.3f}s)")
```

Low-level predictor:
```python
from PIL import Image
from image_evaluator import Hpsv2Predictor

predictor = Hpsv2Predictor(device="cpu", allow_download=True)
img = Image.open("path/to/generated.png")
score = predictor.compute_hpsv2(img, "portrait of a warrior")
print("Score:", score)
```
