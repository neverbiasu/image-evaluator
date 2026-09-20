# CLIP Similarity

## Facts

| Property | Specification |
| :--- | :--- |
| **Category** | Vision-Language Semantic Alignment |
| **CLI Metric** | `clip` |
| **Inputs** | `--image` (file or directory), `--prompt` (text string, file, or directory) |
| **Output** | Raw cosine similarity in `[-1.0, 1.0]` |
| **Direction** | Higher is better |
| **Model** | HuggingFace `openai/clip-vit-base-patch32` |
| **Source** | [Taited/clip-score](https://github.com/Taited/clip-score) / [Hessel et al., 2021](https://arxiv.org/abs/2104.08718) |

## Meaning & Role

CLIP Similarity measures semantic alignment between image embeddings and text prompt embeddings in a shared vision-language latent space. The current CLI computes the raw L2-normalized cosine similarity:

$$S_{\text{CLIP}}(v_{\text{img}}, v_{\text{txt}}) = \frac{v_{\text{img}}}{\|v_{\text{img}}\|_2} \cdot \frac{v_{\text{txt}}}{\|v_{\text{txt}}\|_2}$$

It is primarily designed to evaluate Text-to-Image (T2I) synthesis adherence against descriptive generation prompts (for example, `"a golden retriever running across a sunny lawn"`).

## Scenario Boundaries: T2I Generation vs. Image Editing

CLIP Similarity is designed for **Text-to-Image (T2I) prompt adherence** (evaluating how well an image matches a descriptive scene prompt).

**It is not suitable as an independent metric for image editing evaluation**:
- **Cannot prove editing quality**: Computing CLIP similarity against an editing instruction (e.g. `"make it oil painting"`) or a target description cannot verify source content preservation, local constraints, or artifacts.
- **Editing requires multi-dimensional evaluation**: Evaluating edited images requires pairwise fidelity against the original image (e.g. LPIPS, SSIM, PSNR in [pairwise-fidelity.md](pairwise-fidelity.md)) alongside separate perceptual quality checks.
- **Image-Image comparison is not supported**: The current CLI strictly computes image-text raw cosine similarity. CLIP image-image comparison is not implemented.

## Interpretation & Guidance

1. **No Universal Threshold**: Absolute scores vary by domain and prompt complexity. Do not apply arbitrary cutoffs.
2. **Relative Comparison**: Compare scores only against baseline models or control prompts using the identical `clip-vit-base-patch32` backbone.

## Limitations & Implementation Status

1. Does not measure visual appeal, resolution, facial identity, or complex spatial relations.
2. **Implementation Status**: Evaluates raw L2-normalized cosine similarity without scaling or clipping. Both single-file and folder-level evaluations are covered by unit and regression tests.

## Paper vs. Repository Distinction

- **Paper Definition (Hessel et al., 2021)**: Defines CLIPScore as $2.5 \cdot \max(\cos(e_I, e_T), 0)$, applying a $2.5\times$ scaling factor and zero-clipping.
- **Repository Implementation**: Computes raw L2-normalized cosine similarity $(v_{\text{img}} \cdot v_{\text{txt}})$ directly without scaling or clipping.

## Invocation

```bash
image-evaluator --metrics clip --image path/to/image_or_folder/ --prompt "prompt text or path"

```

## References

- Paper: Hessel et al., 2021, *CLIPScore: A Reference-Free Evaluation Metric for Image Captioning* ([arXiv:2104.08718](https://arxiv.org/abs/2104.08718))
- Base Implementation: [Taited/clip-score](https://github.com/Taited/clip-score)
