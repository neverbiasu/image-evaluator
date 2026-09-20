# ImageReward (ImageReward-v1.0)

## Facts

| Property | Specification |
| :--- | :--- |
| **Category** | Human Preference & Scalar Reward Modeling |
| **CLI Metric** | `image_reward` |
| **Inputs** | `--image` (file or directory), `--prompt` (string, prompt file path, or sequence of prompts) |
| **Output** | Continuous scalar reward score in $\mathbb{R}$ (typical range $[-2.5, 2.5]$, mean $\approx 0$) |
| **Direction** | Higher is better |
| **Model** | BLIP ViT-L vision encoder + Cross-Attention Text Encoder + MLP Reward Head |
| **Backend** | `transformers`, `timm`, `torch` |
| **Weights Repository** | Hugging Face [`THUDM/ImageReward`](https://huggingface.co/THUDM/ImageReward) (`ImageReward.pt`) |
| **Install Extra** | `pip install 'image-evaluator[preference]'` |
| **Citation** | [Xu et al., NeurIPS 2023 (ImageReward)](https://arxiv.org/abs/2304.05977) |

## Meaning & Role

ImageReward evaluates text-to-image generation against human preferences using a multi-modal cross-attention reward model. Unlike dual-tower cosine similarity models, ImageReward directly conditions text representations on image patch embeddings through deep cross-attention layers, and maps the resulting multi-modal representation to a continuous scalar reward value via a learned multi-layer perceptron (MLP):

$$r = \text{MLP}\left( \text{CrossAttention}(\mathbf{z}_{\text{text}}, \mathbf{z}_{\text{image}}) \right)$$
$$S_{\text{ImageReward}}(I, T) = \frac{r - \mu}{\sigma}$$

where $\mu \approx 0.16717$ and $\sigma \approx 1.03334$ are the official standardization constants derived from the training corpus.

## Non-Equivalence Notice: Not a Probability or Cosine Similarity

`image_reward` is **not** a similarity metric, not a probability, and **not** interchangeable with PickScore or HPS v2.1:
- **ImageReward**: Continuous scalar reward output from an MLP regression head. Values can be positive or negative (unbounded real numbers, typically within $[-2.5, +2.5]$, centered near zero). It is neither a cosine similarity nor a bounded probability.
- **PickScore**: Applies temperature-scaled cosine similarity with an MLP classifier head, outputting positive scores typically in $[15.0, 25.0]$.
- **HPS v2.1**: Outputs raw cosine dot products between normalized vision and text vectors, strictly within $[-1.0, 1.0]$ and typically $[0.20, 0.35]$.
- **Never describe ImageReward as similarity or probability**: An ImageReward score of $0.5$ does not mean "50% preference" or "0.5 cosine angle"; it is a standardized scalar reward.

## Architectural Detail & Preprocessing

- **Visual Backbone**: Vision Transformer `ViT-L/16` ($24$ transformer layers, hidden dimension $1024$, $16$ attention heads).
- **Text Backbone**: 12-layer mixture-of-encoders BERT architecture with cross-attention layers attending to the $197$ image patch tokens.
- **Reward Head**: Sequential MLP with 5 linear layers: $768 \to 1024 \to 128 \to 64 \to 16 \to 1$.
- **Preprocessing**: Resize and center-crop to $224 \times 224$, bicubic interpolation, standard ImageNet normalization.
- **Tokenizer**: BERT uncased tokenizer with added special tokens `[DEC]` and `[ENC]`, padded or truncated to $35$ tokens.
- **Memory & Storage**: Checkpoint file is $\approx 1.66$ GiB ($1.78$ GB); CPU evaluation requires $\approx 3.0$ GB RAM.

## Invocation

### CLI Usage

Single image evaluation:
```bash
image-evaluator --metrics image_reward --image path/to/generated.png --prompt "a cute red panda wearing glasses"
```

Directory-level evaluation (broadcast single prompt or map line-by-line prompt file):
```bash
# Broadcast single prompt across folder:
image-evaluator --metrics image_reward --image path/to/folder/ --prompt "a photo of an astronaut on Mars"

# Map one prompt per image from a text file:
image-evaluator --metrics image_reward --image path/to/folder/ --prompt path/to/prompts.txt
```

Explicit opt-in for downloading uncached model weights:
```bash
image-evaluator --metrics image_reward --image generated.png --prompt "sunset over ocean" --allow-download
```

Structured JSON output:
```bash
image-evaluator --metrics image_reward --image generated.png --prompt "cyberpunk city" --format json
```

### Python SDK Usage

High-level evaluation:
```python
from image_evaluator import evaluate, evaluate_detailed

# 1. Basic evaluation
res = evaluate(
    metrics="image_reward",
    image="path/to/generated.png",
    prompt="a digital painting of a mountain landscape",
)
print("ImageReward Score:", res["image_reward"])

# 2. Detailed result
result = evaluate_detailed(
    metrics="image_reward",
    image="path/to/generated.png",
    prompt="a digital painting of a mountain landscape",
)
print(f"ImageReward: {result.scores['image_reward']:.4f} (took {result.duration_seconds:.3f}s)")

# 3. Directory evaluation with line-mapped prompt file or list of prompts
res_folder = evaluate(
    metrics="image_reward",
    image="path/to/folder/",
    prompt=["first image prompt", "second image prompt"],
)
print("Directory ImageReward Mean:", res_folder["image_reward"])
```

Low-level predictor:
```python
from PIL import Image
from image_evaluator import ImageRewardPredictor

predictor = ImageRewardPredictor(device="cpu", allow_download=True)
img = Image.open("path/to/generated.png")
score = predictor.compute_image_reward(img, "portrait of a warrior")
print("Score:", score)
```
