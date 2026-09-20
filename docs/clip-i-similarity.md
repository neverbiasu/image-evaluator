# CLIP-I Image-Image Similarity

## Facts

| Property | Specification |
| :--- | :--- |
| **Category** | Pairwise Visual Semantic & Subject Fidelity |
| **CLI Metric** | `clip_i` |
| **Inputs** | `--image` (file or directory), `--reference` (file or directory) |
| **Output** | Raw cosine similarity in `[-1.0, 1.0]` |
| **Direction** | Higher is better |
| **Model** | OpenAI CLIP `ViT-L/14` with QuickGELU (`ViT-L-14-quickgelu` / `openai`) |
| **Backend** | `open-clip-torch` |
| **Source** | [Radford et al., 2021](https://arxiv.org/abs/2103.00020) / [Ruiz et al., 2023 (DreamBooth)](https://arxiv.org/abs/2208.12242) |

## Meaning & Role

CLIP-I measures the semantic and subject-level visual fidelity between two images (e.g. a reference subject image and a generated or edited image). Unlike pixel-space metrics (PSNR, SSIM) or patch-feature metrics (LPIPS), CLIP-I evaluates visual semantic representation:

$$S_{\text{CLIP-I}}(I_{\text{ref}}, I_{\text{gen}}) = \frac{\phi(I_{\text{ref}})}{\|\phi(I_{\text{ref}})\|_2} \cdot \frac{\phi(I_{\text{gen}})}{\|\phi(I_{\text{gen}})\|_2}$$

where $\phi(I)$ denotes the visual projection embedding extracted by the CLIP ViT-L/14 image encoder.

It is standard in subject-driven generation (such as DreamBooth, Textual Inversion, CustomDiffusion) and image editing benchmarks to evaluate whether the generated subject preserves the visual identity and semantic characteristics of the reference subject.

## Architectural Detail: ViT-L/14 with QuickGELU

Standard CLIP ViT-L/14 models trained by OpenAI employ QuickGELU activations rather than standard GELU. `image-evaluator` explicitly utilizes the exact OpenAI architecture (`ViT-L-14-quickgelu`) to guarantee weight fidelity, preventing activation mismatch and embedding collapse.

Images are automatically resized, bicubically interpolated, and center-cropped to $224 \times 224$ via the standard CLIP visual transform, so identical image dimensions between reference and generated images are not required.

## Invocation

### CLI Usage

Single pair comparison:
```bash
image-evaluator --metrics clip_i --image path/to/generated.png --reference path/to/reference.png
```

Folder-level stem pairing:
```bash
image-evaluator --metrics clip_i --image path/to/generated_dir/ --reference path/to/reference_dir/
```

Allowing external downloads when model weights are uncached:
```bash
image-evaluator --metrics clip_i --image generated.png --reference reference.png --allow-download
```

### Python SDK Usage

```python
from PIL import Image
from image_evaluator import evaluate, evaluate_detailed, ClipIPredictor

# High-level evaluation
results = evaluate(
    metrics="clip_i",
    image="path/to/generated.png",
    reference="path/to/reference.png",
)
print("CLIP-I Score:", results["clip_i"])

# Detailed evaluation with full metadata
result_obj = evaluate_detailed(
    metrics=["clip_i"],
    image=Image.open("path/to/generated.png"),
    reference=Image.open("path/to/reference.png"),
)
print(result_obj.to_json(indent=2))

# Direct predictor instantiation
predictor = ClipIPredictor()
score = predictor.evaluate_clip_i("ref.png", "gen.png")
```

## References

- Radford et al., 2021, *Learning Transferable Visual Models From Natural Language Supervision* ([arXiv:2103.00020](https://arxiv.org/abs/2103.00020))
- Ruiz et al., 2023, *DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation* ([arXiv:2208.12242](https://arxiv.org/abs/2208.12242))

