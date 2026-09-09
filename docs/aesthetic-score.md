# LAION AI Aesthetic Score

## Facts

| Property | Specification |
| :--- | :--- |
| **Category** | Perceptual Visual Quality |
| **CLI Metric** | `aesthetic` |
| **Inputs** | `--image` (file or directory) |
| **Output** | Continuous scalar float |
| **Direction** | Higher is better |
| **Model** | OpenCLIP `ViT-L-14` (`openai` weights, `force_quick_gelu=True`) + linear regression head |
| **Source** | [LAION-AI/aesthetic-predictor](https://github.com/LAION-AI/aesthetic-predictor) |

## Meaning & Role

LAION AI Aesthetic Score estimates perceived visual quality using a linear regression head trained on human ratings from aesthetic datasets (such as SAC and AVA-based scores). It is used for automated aesthetic ranking, quality filtering of synthetic datasets, and assessing the impact of stylistic prompt modifiers.

## Interpretation & Guidance

1. **No Universal Threshold**: Do not use arbitrary cutoff values to categorize images as good or bad.
2. **Relative Comparison**: Compare scores only against control groups or baseline models evaluated with the exact same OpenCLIP `ViT-L-14` backbone.

## Limitations

1. Does not evaluate text-prompt alignment, facial identity, or anatomical correctness.
2. Reflects trained dataset stylistic preferences rather than objective photographic fidelity.

## Implementation Details

`LaionAIAestheticPredictor` extracts 768-dimensional normalized image embeddings using OpenCLIP `ViT-L-14` and applies a linear regression layer (`sa_0_4_vit_l_14_linear.pth`). Folder evaluation computes the arithmetic mean across all valid image files.

### Activation Function & Score Shift

The predictor explicitly passes `force_quick_gelu=True` during OpenCLIP model instantiation:

1. **Alignment with Upstream Weights & Reference Implementation**: Setting `force_quick_gelu=True` aligns the model configuration with the official OpenAI pre-trained weights and the traced historical reference implementation (`open_clip==1.3.0` loaded OpenAI TorchScript weights directly, executing QuickGELU). Subsequent `open_clip` releases defaulted to standard PyTorch `nn.GELU()` for `ViT-L-14`, triggering a configuration mismatch warning against the `openai` tag. Because the exact training script for the linear head (`sa_0_4_vit_l_14_linear.pth`) is unavailable in upstream records, this migration does not claim to have fully reconstructed the original training configuration; rather, it ensures architectural consistency with the pre-trained weights and eliminates the upstream warning.
2. **Score Shift**: Switching from standard GELU fallback to QuickGELU alters the extracted visual embeddings and shifts individual predicted scores (e.g., shifts between approximately -0.20 and +0.04 observed on reference test cases).
3. **No Claim of Inherent Quality Improvement**: This activation alignment restores consistency with the upstream reference implementation and removes the upstream warning, but does not represent a claim that predicted scoring quality or accuracy is intrinsically improved.

## Invocation

```bash
image-evaluator --metrics aesthetic --image path/to/image_or_folder/

```

## References

- Repository: [LAION-AI/aesthetic-predictor](https://github.com/LAION-AI/aesthetic-predictor)
- Checkpoint: `sa_0_4_vit_l_14_linear.pth`
