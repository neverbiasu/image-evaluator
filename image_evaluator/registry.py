"""Read-only catalog of image evaluation metric specifications."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from image_evaluator.specifications import (
    ImplementationRef,
    InputContract,
    MetricSpec,
)


class DuplicateMetricError(ValueError):
    """Raised when a registry is built with a repeated metric ID."""


class UnknownMetricError(LookupError):
    """Raised when a requested metric ID is not registered."""


@dataclass(frozen=True, slots=True)
class MetricRegistry:
    """An immutable, metadata-only metric catalog."""

    _metrics: Mapping[str, MetricSpec]

    def __init__(self, metrics: Iterable[MetricSpec]) -> None:
        indexed: dict[str, MetricSpec] = {}
        for metric in metrics:
            if metric.id in indexed:
                raise DuplicateMetricError(
                    f"Metric id '{metric.id}' is already registered."
                )
            indexed[metric.id] = metric
        object.__setattr__(self, "_metrics", MappingProxyType(indexed))

    def list(self) -> tuple[MetricSpec, ...]:
        """Return every metric ordered by canonical ID."""

        return tuple(self._metrics[key] for key in sorted(self._metrics))

    def get(self, metric_id: str) -> MetricSpec:
        """Return one metric or raise an explicit unknown-ID error."""

        try:
            return self._metrics[metric_id]
        except KeyError as exc:
            raise UnknownMetricError(
                f"Unknown metric id '{metric_id}'. Available metrics: "
                f"{sorted(self._metrics)}"
            ) from exc

    def filter(
        self,
        *,
        task: str | None = None,
        objective: str | None = None,
    ) -> tuple[MetricSpec, ...]:
        """Return metrics matching both supplied discovery terms."""

        return tuple(
            metric
            for metric in self.list()
            if (task is None or task in metric.tasks)
            and (objective is None or objective in metric.objectives)
        )


_IMAGE = InputContract(required=("image",))
_IMAGE_PROMPT = InputContract(required=("image", "prompt"))
_PAIR = InputContract(required=("image", "reference_image"))
_DISTRIBUTION = InputContract(
    required=("image_collection", "reference_collection")
)

METRIC_REGISTRY = MetricRegistry(
    (
        MetricSpec(
            id="aesthetic",
            display_name="LAION Aesthetic Score",
            tasks=("text_to_image", "image_editing"),
            objectives=("aesthetic_quality",),
            inputs=_IMAGE,
            score_direction="higher_is_better",
            implementation=ImplementationRef(
                backend="laion-aesthetic-predictor",
                protocol="linear-head-on-clip-embedding",
                model="vit_l_14",
            ),
            aggregation=("arithmetic_mean_for_directory_inputs",),
            dependencies=("open-clip-torch", "torch"),
            citations=(
                "https://github.com/LAION-AI/aesthetic-predictor",
            ),
            docs_path="docs/aesthetic-score.md",
        ),
        MetricSpec(
            id="arcface",
            display_name="ArcFace Distance",
            tasks=("face_generation", "face_editing"),
            objectives=("identity_preservation",),
            inputs=_PAIR,
            score_direction="lower_is_better",
            implementation=ImplementationRef(
                backend="insightface",
                protocol="face-embedding-distance",
                model="buffalo_l",
            ),
            aggregation=("arithmetic_mean_for_directory_inputs",),
            dependencies=("insightface", "onnxruntime"),
            citations=(
                "https://arxiv.org/abs/1801.07698",
                "https://github.com/deepinsight/insightface",
            ),
            docs_path="docs/arcface-distance.md",
        ),
        MetricSpec(
            id="clip",
            display_name="CLIP Score",
            tasks=("text_to_image", "image_editing"),
            objectives=("text_image_alignment",),
            inputs=_IMAGE_PROMPT,
            score_direction="higher_is_better",
            implementation=ImplementationRef(
                backend="transformers",
                protocol="image-text-cosine-similarity",
                model="openai/clip-vit-base-patch32",
            ),
            aggregation=("arithmetic_mean_for_directory_inputs",),
            dependencies=("transformers", "torch"),
            citations=(
                "https://arxiv.org/abs/2104.08718",
                "https://github.com/Taited/clip-score",
            ),
            docs_path="docs/clip-similarity.md",
        ),
        MetricSpec(
            id="directional_clip",
            display_name="Directional CLIP",
            tasks=("image_editing",),
            objectives=("edit_direction_alignment",),
            inputs=InputContract(
                required=(
                    "image",
                    "reference_image",
                    "prompt",
                    "source_prompt",
                )
            ),
            score_direction="higher_is_better",
            implementation=ImplementationRef(
                backend="transformers",
                protocol="directional-embedding-cosine-similarity",
                model="openai/clip-vit-base-patch32",
            ),
            dependencies=("transformers", "torch"),
            citations=(
                "https://arxiv.org/abs/2108.00946",
                "https://arxiv.org/abs/2211.09800",
            ),
            docs_path="docs/directional-clip.md",
        ),
        MetricSpec(
            id="fid",
            display_name="Fréchet Inception Distance",
            tasks=("text_to_image",),
            objectives=("distribution_similarity",),
            inputs=_DISTRIBUTION,
            score_direction="lower_is_better",
            implementation=ImplementationRef(
                backend="clean-fid",
                protocol="clean-fid",
                model="inception_v3",
                backend_version="0.1.35",
            ),
            dependencies=("clean-fid",),
            citations=(
                "Parmar et al., On Aliased Resizing and Surprising "
                "Subtleties in GAN Evaluation, CVPR 2022",
            ),
            docs_path="docs/fid.md",
        ),
        MetricSpec(
            id="kid",
            display_name="Kernel Inception Distance",
            tasks=("text_to_image",),
            objectives=("distribution_similarity",),
            inputs=_DISTRIBUTION,
            score_direction="lower_is_better",
            implementation=ImplementationRef(
                backend="clean-fid",
                protocol="polynomial-mmd",
                model="inception_v3",
                backend_version="0.1.35",
            ),
            aggregation=("mean_over_random_subsets",),
            dependencies=("clean-fid",),
            citations=(
                "Binkowski et al., Demystifying MMD GANs, ICLR 2018",
                "Parmar et al., On Aliased Resizing and Surprising "
                "Subtleties in GAN Evaluation, CVPR 2022",
            ),
            docs_path="docs/kid.md",
        ),
        MetricSpec(
            id="lpips",
            display_name="Learned Perceptual Image Patch Similarity",
            tasks=(
                "image_editing",
                "image_reconstruction",
                "super_resolution",
            ),
            objectives=("perceptual_similarity",),
            inputs=_PAIR,
            score_direction="lower_is_better",
            implementation=ImplementationRef(
                backend="lpips",
                protocol="deep-feature-distance",
                model="alex",
            ),
            aggregation=("arithmetic_mean_for_directory_inputs",),
            dependencies=("lpips", "torch"),
            citations=(
                "https://github.com/richzhang/PerceptualSimilarity",
            ),
            docs_path="docs/pairwise-fidelity.md",
        ),
        MetricSpec(
            id="pickscore",
            display_name="PickScore",
            tasks=("text_to_image", "image_editing"),
            objectives=("human_preference",),
            inputs=_IMAGE_PROMPT,
            score_direction="higher_is_better",
            implementation=ImplementationRef(
                backend="transformers",
                protocol="prompt-conditioned-preference-score",
                model="yuvalkirstain/PickScore_v1",
            ),
            aggregation=("arithmetic_mean_for_directory_inputs",),
            dependencies=("transformers", "torch"),
            citations=(
                "Kirstain et al., Pick-a-Pic: An Open Dataset of User "
                "Preferences for Text-to-Image Generation, NeurIPS 2023",
            ),
            docs_path="docs/pickscore.md",
        ),
        MetricSpec(
            id="psnr",
            display_name="Peak Signal-to-Noise Ratio",
            tasks=(
                "image_editing",
                "image_reconstruction",
                "super_resolution",
            ),
            objectives=("pixel_fidelity",),
            inputs=_PAIR,
            score_direction="higher_is_better",
            implementation=ImplementationRef(
                backend="image-evaluator",
                protocol="peak-signal-to-noise-ratio",
            ),
            aggregation=("arithmetic_mean_for_directory_inputs",),
            dependencies=("torch",),
            citations=("Standard signal processing definition",),
            docs_path="docs/pairwise-fidelity.md",
        ),
        MetricSpec(
            id="ssim",
            display_name="Structural Similarity Index Measure",
            tasks=(
                "image_editing",
                "image_reconstruction",
                "super_resolution",
            ),
            objectives=("structural_similarity",),
            inputs=_PAIR,
            score_direction="higher_is_better",
            implementation=ImplementationRef(
                backend="image-evaluator",
                protocol="structural-similarity-index",
            ),
            aggregation=("arithmetic_mean_for_directory_inputs",),
            dependencies=("torch",),
            citations=("Wang et al., IEEE TIP 2004",),
            docs_path="docs/pairwise-fidelity.md",
        ),
    )
)


def list_metrics() -> tuple[MetricSpec, ...]:
    """List all registered metrics without loading evaluation backends."""

    return METRIC_REGISTRY.list()


def get_metric(metric_id: str) -> MetricSpec:
    """Get a registered metric without loading its evaluation backend."""

    return METRIC_REGISTRY.get(metric_id)


def filter_metrics(
    *, task: str | None = None, objective: str | None = None
) -> tuple[MetricSpec, ...]:
    """Filter metrics by open task and objective vocabulary terms."""

    return METRIC_REGISTRY.filter(task=task, objective=objective)
