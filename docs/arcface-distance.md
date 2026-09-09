# ArcFace Distance

## Facts

| Property | Specification |
| :--- | :--- |
| **Category** | Facial Identity Preservation |
| **CLI Metric** | `arcface` |
| **Inputs** | `--image` (generated image/folder), `--reference` (reference face/folder) |
| **Output** | Cosine distance in `[0.0, 2.0]` |
| **Direction** | Lower is better |
| **Model** | InsightFace `buffalo_l` |
| **Source** | [deepinsight/insightface](https://github.com/deepinsight/insightface) / [Deng et al., 2019](https://arxiv.org/abs/1801.07698) |

## Meaning & Role

ArcFace Distance measures identity dissimilarity between a generated face and a reference photo by computing the cosine distance of their deep facial feature embeddings. It is used to evaluate identity retention in personalization methods like LoRA, InstantID, and DreamBooth.

## Interpretation & Guidance

1. **Mathematical Scale**: Cosine distance is defined as $1 - \cos(u, v)$. A value of `0.0` represents identical feature vectors; lower values indicate greater facial similarity.
2. **No Universal Threshold**: Verification thresholds require empirical calibration on specific test benchmarks. Do not apply uncalibrated universal thresholds.
3. **Protocol Consistency**: Compare distances only against reference pairs evaluated under identical InsightFace model packs and alignment pipelines.

## Limitations

1. Evaluates only facial biometric features; ignores background, clothing, and overall image quality.
2. Requires successful face detection in both reference and target images; returns `None` if detection fails.
3. Folder evaluation pairs images strictly by alphabetical sort order (`zip(sorted(reference), sorted(generated))`).

## Implementation Details

`ArcFaceDistPredictor` uses `FaceAnalysis("buffalo_l")` to extract 512-dimensional embeddings from the primary detected face (`faces[0].embedding`) and computes cosine distance via PyTorch.

## Invocation

```bash
image-evaluator --metrics arcface --image path/to/image_or_folder/ --reference path/to/ref_or_folder/

```

## References

- Paper: Deng et al., 2019, *ArcFace: Additive Angular Margin Loss for Deep Face Recognition* ([arXiv:1801.07698](https://arxiv.org/abs/1801.07698))
- Implementation: [deepinsight/insightface](https://github.com/deepinsight/insightface)
