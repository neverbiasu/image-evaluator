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

CLIP Similarity measures semantic alignment between generated image embeddings and text prompt embeddings in a shared vision-language space. It is used to benchmark text-to-image prompt adherence and evaluate prompt engineering strategies.

## Interpretation & Guidance

1. **No Universal Threshold**: Absolute scores vary by domain and prompt complexity. Do not apply arbitrary cutoffs.
2. **Relative Comparison**: Compare scores only against baseline models or control prompts using the identical `clip-vit-base-patch32` backbone.

## Limitations & Implementation Status

1. Does not measure visual appeal, resolution, facial identity, or complex spatial relations.
2. **Implementation Status**: Single-file path expansion (`_combine_without_prefix`) and tensor modality routing are fixed and covered by 8 mocked unit tests. Real-model end-to-end integration validation remains scheduled for M2.

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
