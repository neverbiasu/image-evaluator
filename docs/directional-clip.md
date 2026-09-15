# Directional CLIP (ΔCLIP)

## Facts

| Property | Specification |
| :--- | :--- |
| **Category** | Image Editing Semantic Direction Alignment |
| **CLI Metric** | `directional_clip` |
| **Inputs** | `--image` (edited image), `--reference` (source image), `--prompt` (target prompt), `--prompt-src` (source prompt) |
| **Output** | Directional cosine similarity in `[-1.0, 1.0]` |
| **Direction** | Higher is better |
| **Model** | HuggingFace `openai/clip-vit-base-patch32` |
| **Source** | [Gal et al., StyleGAN-NADA (SIGGRAPH 2022)](https://arxiv.org/abs/2108.00946) / [Brooks et al., InstructPix2Pix (CVPR 2023)](https://arxiv.org/abs/2211.09800) |

## Meaning & Role

Standard CLIP score measures static semantic alignment between an image and a text prompt, but cannot capture whether an *edit* correctly changed an image in the intended direction. Directional CLIP resolves this by measuring the alignment between the image change vector and text change vector in CLIP joint embedding space:

$$\Delta I = \frac{E_I(I_{\text{edit}}) - E_I(I_{\text{src}})}{\|E_I(I_{\text{edit}}) - E_I(I_{\text{src}})\|_2}$$

$$\Delta T = \frac{E_T(T_{\text{target}}) - E_T(T_{\text{source}})}{\|E_T(T_{\text{target}}) - E_T(T_{\text{source}})\|_2}$$

$$\text{DirectionalScore} = \Delta I \cdot \Delta T$$

## Four-Input Contract

Evaluating image editing requires four explicit inputs:
1. `image_src`: Source image before editing (`--reference` in CLI).
2. `image_edit`: Output image after editing (`--image` in CLI).
3. `prompt_src`: Text description of the source image (`--prompt-src` in CLI).
4. `prompt_target`: Text description of the intended edit target (`--prompt` in CLI).

## Interpretation & Guidance

1. **Continuous Range**: Scores fall in $[-1.0, 1.0]$. Positive values indicate that the image change aligns with the text instruction; negative values indicate that the change moved opposite to the instruction.
2. **Relative Ranking**: In controlled benchmark testing, Directional CLIP achieves 100% relative ranking accuracy distinguishing positive edits from negative/unrelated edits.
3. **Numerical Safeguards**: Built-in near-zero delta protection ($\epsilon=10^{-6}$) returns `0.0` when the image or text difference vector is degenerate, preventing NaN values.
4. **Multimodal Evaluation**: Best combined with pairwise fidelity metrics (LPIPS/SSIM) to ensure background preservation while tracking edit magnitude.
