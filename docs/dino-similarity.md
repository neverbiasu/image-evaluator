# DINOv2 Image-Image Similarity

## Facts

| Property | Specification |
| :--- | :--- |
| **Category** | Pairwise Visual Semantic & Structural Fidelity |
| **CLI Metric** | `dino_similarity` |
| **Inputs** | `--image` (file or directory), `--reference` (file or directory) |
| **Output** | Raw cosine similarity in `[-1.0, 1.0]` |
| **Direction** | Higher is better |
| **Model** | HuggingFace `facebook/dinov2-base` (Vision Transformer Base, patch 14) |
| **Backend** | `transformers` |
| **Source** | [Oquab et al., 2023 (DINOv2)](https://arxiv.org/abs/2304.07193) |

## Meaning & Role

DINOv2 Image-Image Similarity measures visual structural and semantic fidelity between paired images by comparing normalized representations extracted by a self-supervised Vision Transformer. Unlike supervised classifiers or text-aligned encoders (CLIP), DINOv2 learns rich spatial and geometric representations without text supervision:

$$S_{\text{DINO}}(I_{\text{ref}}, I_{\text{gen}}) = \frac{\mathbf{z}_{\text{cls}}(I_{\text{ref}})}{\|\mathbf{z}_{\text{cls}}(I_{\text{ref}})\|_2} \cdot \frac{\mathbf{z}_{\text{cls}}(I_{\text{gen}})}{\|\mathbf{z}_{\text{cls}}(I_{\text{gen}})\|_2}$$

where $\mathbf{z}_{\text{cls}}(I)$ denotes the L2-normalized CLS token embedding from the final layer of the `facebook/dinov2-base` backbone.

In generative image evaluation (such as instruction-guided image editing and personalized generation benchmarks), DINOv2 similarity serves as a sensitive measure of structural coherence, object layout preservation, and low-level visual semantic fidelity.

## Non-Equivalence Notice: Not PIE-Bench Structure Distance

Generic DINO cosine similarity is **not** equivalent to the specialized DINO structure distance defined in certain benchmark suites (such as PIE-Bench):
- **PIE-Bench Structure Distance**: Computes self-attention map differences across deep transformer layers or spatial patch feature correlations under specific masked foreground regions.
- **Repository Implementation**: Evaluates global final CLS-token cosine similarity across the entire image field without spatial mask slicing or multi-head attention extraction.

## Architectural Detail & Preprocessing

`facebook/dinov2-base` operates with a frozen patch size of 14:
- Preprocessor resizes and center-crops input images to $224 \times 224$ pixels, normalizes with ImageNet mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`.
- Identical image dimensions between reference and generated images are not required.

## Invocation

### CLI Usage

Single pair comparison:
```bash
image-evaluator --metrics dino_similarity --image path/to/generated.png --reference path/to/reference.png
```

Folder-level stem pairing:
```bash
image-evaluator --metrics dino_similarity --image path/to/generated_dir/ --reference path/to/reference_dir/
```

Allowing external downloads when model weights are uncached:
```bash
image-evaluator --metrics dino_similarity --image generated.png --reference reference.png --allow-download
```

### Python SDK Usage

```python
from PIL import Image
from image_evaluator import evaluate, evaluate_detailed, DinoSimilarityPredictor

# High-level evaluation
results = evaluate(
    metrics="dino_similarity",
    image="path/to/generated.png",
    reference="path/to/reference.png",
)
print("DINOv2 Score:", results["dino_similarity"])

# Detailed evaluation with full metadata
result_obj = evaluate_detailed(
    metrics=["dino_similarity"],
    image=Image.open("path/to/generated.png"),
    reference=Image.open("path/to/reference.png"),
)
print(result_obj.to_json(indent=2))

# Direct predictor instantiation
predictor = DinoSimilarityPredictor()
score = predictor.evaluate_dino_similarity("ref.png", "gen.png")
```

## References

- Oquab et al., 2023, *DINOv2: Learning Robust Visual Features without Supervision* ([arXiv:2304.07193](https://arxiv.org/abs/2304.07193))

