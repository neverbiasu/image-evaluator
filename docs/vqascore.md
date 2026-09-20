# VQAScore (CLIP-FlanT5-XL 3B)

## Facts

| Property | Specification |
| :--- | :--- |
| **Category** | Visual Question Answering Alignment & Compositionality |
| **CLI Metric** | `vqascore` |
| **Inputs** | `--image` (file or directory), `--prompt` (string) |
| **Output** | Posterior probability $P(\text{"Yes"} \mid \text{Image}, \text{Text}) \in [0.0, 1.0]$ |
| **Direction** | Higher is better |
| **Model** | Fine-tuned CLIP-FlanT5-XL (3B) + CLIP-ViT-L/14-336 vision tower |
| **Backend** | `transformers`, `accelerate`, `sentencepiece`, `torch` |
| **Weights Repository** | Hugging Face [`zhiqiulin/clip-flant5-xl`](https://huggingface.co/zhiqiulin/clip-flant5-xl) + [`openai/clip-vit-large-patch14-336`](https://huggingface.co/openai/clip-vit-large-patch14-336) |
| **Install Extra** | `pip install 'image-evaluator[vqa]'` |
| **Citation** | [Lin et al., ECCV 2024 (VQAScore)](https://arxiv.org/abs/2404.01291); [Lin et al., CVPR 2024 (GenAI-Bench)](https://arxiv.org/abs/2406.13743) |

## Meaning & Role

VQAScore evaluates text-to-visual generation by framing image-text alignment as visual question answering (VQA). Conventional dual-encoder models (such as CLIPScore) project whole images and prompts into isolated feature vectors, frequently suffering from "bag-of-words" degradation—ignoring negation, relative spatial relationships, count, and multi-attribute binding.

VQAScore overcomes this by formulating generation evaluation as the posterior probability that a visual-language model answers **"Yes"** to whether the image depicts the prompt:

$$\text{Question: Does this figure show "{prompt}"? Please answer yes or no.}$$
$$\text{Answer: Yes}$$

The evaluation score is computed directly from the sequence cross-entropy loss over the target tokens:

$$\text{VQAScore}(I, T) = P(\text{"Yes"} \mid I, T) = \exp\left(-\mathcal{L}_{\text{CE}}(\hat{\mathbf{y}}, \mathbf{y}_{\text{"Yes"}})\right)$$

This metric produces a continuous, calibrated probability strictly bounded in $[0.0, 1.0]$, where higher values indicate stronger text-image alignment and compositional fidelity.

## Non-Equivalence Notice: Not a Cosine Similarity or Scalar Reward

`vqascore` is **not** a cosine similarity metric and **not** an unbounded scalar reward:
- **VQAScore**: Formulated as a generation posterior probability in $[0.0, 1.0]$. It evaluates explicit question answering accuracy and fine-grained visual compositionality.
- **CLIPScore / CLIP-I**: Evaluates cosine similarity between dual-encoder projection vectors in $[-1.0, 1.0]$.
- **PickScore**: Measures paired preference via temperature-scaled sigmoid logits, typically producing values in $[15.0, 25.0]$.
- **ImageReward**: Uses an MLP regression head on top of cross-attention features, outputting an unbounded scalar reward in $\mathbb{R}$ (typically $[-2.5, 2.5]$).
- **HPS v2.1**: Evaluates fine-tuned cosine similarity between normalized visual and text representations.

## Architectural Detail & Preprocessing

- **Visual Backbone**: Frozen `openai/clip-vit-large-patch14-336` vision tower, extracting patch hidden states from the penultimate layer (layer $-2$, 576 patch tokens).
- **Vision-Language Projector**: Two-layer MLP projector (`mlp2x_gelu`): $\text{Linear}(1024 \to 2048) \to \text{GELU} \to \text{Linear}(2048 \to 2048)$.
- **Language Backbone**: Google `google/flan-t5-xl` sequence-to-sequence model ($3\text{B}$ parameters), fine-tuned on multi-modal instruction data.
- **Precision & Execution**: Loaded in `torch.bfloat16` on CPU / Apple Silicon MPS or `torch.float16` on CUDA GPU.
- **Aspect Ratio Preservation**: Applies `expand2square` padding with the CLIP preprocessor image mean color before resizing, eliminating distortive image stretching.
- **Memory Footprint**: Total download is $\approx 7.49\text{ GiB}$ across the two Hugging Face snapshot repositories. In-memory RSS footprint on CPU is $\approx 1.83\text{ GiB}$, enabling safe execution on 16GB developer machines.

## Invocation

### CLI Usage

Single image evaluation:
```bash
image-evaluator --metrics vqascore --image path/to/image.png --prompt "a red sports car parked in front of a modern building"
```

Directory-level evaluation (arithmetic mean across all valid images):
```bash
image-evaluator --metrics vqascore --image path/to/folder/ --prompt "a golden retriever playing in autumn leaves"
```

Explicit download authorization for first-time uncached execution:
```bash
image-evaluator --metrics vqascore --image test.png --prompt "a blue bicycle" --allow-download
```

Structured JSON output:
```bash
image-evaluator --metrics vqascore --image test.png --prompt "a coffee mug on a wooden desk" --format json
```

### Python SDK Usage

High-level evaluation:
```python
from image_evaluator import evaluate, evaluate_detailed

# 1. Basic dictionary evaluation
res = evaluate(
    metrics="vqascore",
    image="path/to/image.png",
    prompt="a red sports car parked in front of a modern building",
    allow_download=True,
)
print("VQAScore:", res["vqascore"])

# 2. Detailed result object
detailed_res = evaluate_detailed(
    metrics="vqascore",
    image="path/to/image.png",
    prompt="a red sports car parked in front of a modern building",
)
print(f"VQAScore: {detailed_res.scores['vqascore']:.6f} "
      f"(duration: {detailed_res.duration_seconds:.2f}s)")
```

Low-level predictor:
```python
from PIL import Image
from image_evaluator import VQAScorePredictor

predictor = VQAScorePredictor(device="cpu", allow_download=True)
img = Image.open("path/to/image.png")
score = predictor.compute_vqascore(img, "a red sports car parked in front of a modern building")
print("VQAScore:", score)
```
