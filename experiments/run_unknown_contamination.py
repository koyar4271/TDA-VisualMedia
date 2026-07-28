from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
import yaml


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

from run_adverse_ordering import (
    compute_negative_cache_logits,
    compute_positive_cache_logits,
    flatten_cache,
    load_torch_artifact,
    update_cache,
)


@dataclass
class TrackedCacheEntry:
    entry_id: int
    source_index: int
    origin: str
    feature: torch.Tensor
    entropy: float
    probability_map: torch.Tensor | None
    pseudo_label: int
    true_label: int
    insertion_step: int

    @property
    def is_unknown(self) -> bool:
        return self.origin == "unknown"

    @property
    def is_correct(self) -> bool:
        return (
            self.origin == "known"
            and self.pseudo_label == self.true_label
        )


@dataclass(frozen=True)
class StreamItem:
    origin: str
    source_index: int
    known_position: int | None


@dataclass
class SimulationResult:
    sample_df: pd.DataFrame
    unknown_event_df: pd.DataFrame
    counters: dict[str, int]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate TDA under unknown-class cache contamination."
    )
    parser.add_argument("--known-features", required=True)
    parser.add_argument("--unknown-features", required=True)
    parser.add_argument("--config", default="./configs/caltech101.yaml")
    parser.add_argument("--output-root", default="./results")
    parser.add_argument("--known-order-seed", type=int, default=0)
    parser.add_argument(
        "--prefix-sizes",
        type=int,
        nargs="+",
        default=[100, 500],
    )
    parser.add_argument(
        "--unknown-counts",
        type=int,
        nargs="+",
        default=[0, 25, 50, 100, 200],
    )
    parser.add_argument(
        "--unknown-seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
    )
    parser.add_argument("--recovery-bin-size", type=int, default=50)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--main-prefix-size", type=int, default=100)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def validate_artifacts(
    known_artifact: dict[str, Any],
    unknown_artifact: dict[str, Any],
) -> None:
    common_keys = [
        "backbone",
        "classnames",
        "num_classes",
        "features",
        "clip_logits",
        "probabilities",
        "raw_entropy",
        "normalized_entropy",
        "clip_predictions",
        "max_similarity",
        "probability_margin",
        "image_paths",
    ]

    for artifact_name, artifact in [
        ("known", known_artifact),
        ("unknown", unknown_artifact),
    ]:
        missing = [key for key in common_keys if key not in artifact]
        if missing:
            raise KeyError(
                f"The {artifact_name} artifact is missing keys: "
                + ", ".join(missing)
            )

    if known_artifact["backbone"] != unknown_artifact["backbone"]:
        raise ValueError(
            "Known and unknown features use different CLIP backbones."
        )

    if list(known_artifact["classnames"]) != list(
        unknown_artifact["classnames"]
    ):
        raise ValueError(
            "Known and unknown artifacts use different class names."
        )

    if int(known_artifact["num_classes"]) != int(
        unknown_artifact["num_classes"]
    ):
        raise ValueError(
            "Known and unknown artifacts use different class counts."
        )


def filter_cache_by_origin(
    cache: dict[int, list[TrackedCacheEntry]],
    origin: str,
) -> dict[int, list[TrackedCacheEntry]]:
    filtered: dict[int, list[TrackedCacheEntry]] = {}

    for class_index, entries in cache.items():
        selected = [entry for entry in entries if entry.origin == origin]
        if selected:
            filtered[class_index] = selected

    return filtered


def cache_contamination_ratio(
    cache: dict[int, list[TrackedCacheEntry]],
) -> float:
    entries = flatten_cache(cache)
    if not entries:
        return 0.0
    return sum(entry.is_unknown for entry in entries) / len(entries)


def cache_unknown_count(
    cache: dict[int, list[TrackedCacheEntry]],
) -> int:
    return sum(entry.is_unknown for entry in flatten_cache(cache))


def zero_logits(
    num_classes: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.zeros(
        (1, num_classes),
        device=device,
        dtype=dtype,
    )


def compute_origin_split_logits(
    query_feature: torch.Tensor,
    positive_cache: dict[int, list[TrackedCacheEntry]],
    negative_cache: dict[int, list[TrackedCacheEntry]],
    config: dict[str, Any],
    num_classes: int,
) -> dict[str, torch.Tensor]:
    pos_cfg = config["positive"]
    neg_cfg = config["negative"]
    dtype = query_feature.dtype
    device = query_feature.device

    outputs = {
        "positive_known": zero_logits(num_classes, device, dtype),
        "positive_unknown": zero_logits(num_classes, device, dtype),
        "negative_known": zero_logits(num_classes, device, dtype),
        "negative_unknown": zero_logits(num_classes, device, dtype),
    }

    if bool(pos_cfg["enabled"]):
        for origin in ["known", "unknown"]:
            selected_cache = filter_cache_by_origin(
                positive_cache,
                origin,
            )
            if selected_cache:
                outputs[f"positive_{origin}"] = (
                    compute_positive_cache_logits(
                        query_feature,
                        selected_cache,
                        float(pos_cfg["alpha"]),
                        float(pos_cfg["beta"]),
                        num_classes,
                    )
                )

    if bool(neg_cfg["enabled"]):
        for origin in ["known", "unknown"]:
            selected_cache = filter_cache_by_origin(
                negative_cache,
                origin,
            )
            if selected_cache:
                outputs[f"negative_{origin}"] = (
                    compute_negative_cache_logits(
                        query_feature,
                        selected_cache,
                        float(neg_cfg["alpha"]),
                        float(neg_cfg["beta"]),
                        num_classes,
                        float(neg_cfg["mask_threshold"]["lower"]),
                        float(neg_cfg["mask_threshold"]["upper"]),
                    )
                )

    return outputs


def finalize_unknown_event(
    event_rows: list[dict[str, Any]],
    run_name: str,
    prefix_size: int,
    unknown_count: int,
    unknown_seed: int,
    cache_type: str,
    entry: TrackedCacheEntry,
    removal_step: int,
    censored: bool,
) -> None:
    if not entry.is_unknown:
        return

    event_rows.append(
        {
            "run_name": run_name,
            "prefix_size": prefix_size,
            "unknown_count": unknown_count,
            "unknown_seed": unknown_seed,
            "cache_type": cache_type,
            "entry_id": entry.entry_id,
            "source_index": entry.source_index,
            "pseudo_label": entry.pseudo_label,
            "insertion_step": entry.insertion_step,
            "removal_step": removal_step,
            "lifetime": removal_step - entry.insertion_step,
            "censored": int(censored),
        }
    )


def get_sample_tensors(
    artifact: dict[str, Any],
    source_index: int,
    device: torch.device,
) -> dict[str, Any]:
    return {
        "feature": artifact["features"][source_index]
        .to(device=device)
        .unsqueeze(0),
        "clip_logits": artifact["clip_logits"][source_index]
        .to(device=device)
        .unsqueeze(0),
        "probability_map": artifact["probabilities"][source_index].to(
            device=device
        ),
        "raw_entropy": float(artifact["raw_entropy"][source_index]),
        "normalized_entropy": float(
            artifact["normalized_entropy"][source_index]
        ),
        "clip_prediction": int(
            artifact["clip_predictions"][source_index]
        ),
        "max_similarity": float(
            artifact["max_similarity"][source_index]
        ),
        "probability_margin": float(
            artifact["probability_margin"][source_index]
        ),
        "image_path": str(artifact["image_paths"][source_index]),
    }


def build_known_order(sample_count: int, seed: int) -> list[int]:
    order = list(range(sample_count))
    random.Random(seed).shuffle(order)
    return order


def select_unknown_indices(
    sample_count: int,
    unknown_count: int,
    seed: int,
) -> list[int]:
    if unknown_count > sample_count:
        raise ValueError(
            f"Requested {unknown_count} unknown samples, but only "
            f"{sample_count} are available."
        )

    return random.Random(seed).sample(
        range(sample_count),
        unknown_count,
    )


def build_control_stream(known_order: list[int]) -> list[StreamItem]:
    return [
        StreamItem(
            origin="known",
            source_index=source_index,
            known_position=known_position,
        )
        for known_position, source_index in enumerate(known_order)
    ]


def build_contaminated_stream(
    known_order: list[int],
    prefix_size: int,
    unknown_indices: Iterable[int],
) -> list[StreamItem]:
    prefix = [
        StreamItem(
            origin="known",
            source_index=source_index,
            known_position=known_position,
        )
        for known_position, source_index in enumerate(
            known_order[:prefix_size]
        )
    ]
    unknown_block = [
        StreamItem(
            origin="unknown",
            source_index=source_index,
            known_position=None,
        )
        for source_index in unknown_indices
    ]
    suffix = [
        StreamItem(
            origin="known",
            source_index=source_index,
            known_position=known_position,
        )
        for known_position, source_index in enumerate(
            known_order[prefix_size:],
            start=prefix_size,
        )
    ]
    return prefix + unknown_block + suffix


def simulate_stream(
    known_artifact: dict[str, Any],
    unknown_artifact: dict[str, Any],
    config: dict[str, Any],
    stream: list[StreamItem],
    run_name: str,
    prefix_size: int,
    unknown_count: int,
    unknown_seed: int,
    device: torch.device,
) -> SimulationResult:
    num_classes = int(known_artifact["num_classes"])
    classnames = list(known_artifact["classnames"])
    labels = known_artifact["labels"]

    pos_cfg = config["positive"]
    neg_cfg = config["negative"]

    positive_cache: dict[int, list[TrackedCacheEntry]] = {}
    negative_cache: dict[int, list[TrackedCacheEntry]] = {}
    sample_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []

    next_positive_entry_id = 0
    next_negative_entry_id = 0
    counters = {
        "unknown_positive_admission_count": 0,
        "unknown_negative_admission_count": 0,
        "known_positive_admission_count": 0,
        "known_negative_admission_count": 0,
    }

    for stream_step, item in enumerate(stream):
        artifact = (
            known_artifact
            if item.origin == "known"
            else unknown_artifact
        )
        sample = get_sample_tensors(
            artifact,
            item.source_index,
            device,
        )

        target = (
            int(labels[item.source_index])
            if item.origin == "known"
            else -1
        )
        true_class = (
            classnames[target]
            if item.origin == "known"
            else "UNKNOWN"
        )

        positive_admitted = False
        negative_admitted = False
        positive_replaced = False
        negative_replaced = False

        if bool(pos_cfg["enabled"]):
            positive_entry = TrackedCacheEntry(
                entry_id=next_positive_entry_id,
                source_index=item.source_index,
                origin=item.origin,
                feature=sample["feature"].squeeze(0),
                entropy=sample["raw_entropy"],
                probability_map=None,
                pseudo_label=sample["clip_prediction"],
                true_label=target,
                insertion_step=stream_step,
            )
            positive_admitted, removed_entry = update_cache(
                positive_cache,
                positive_entry,
                int(pos_cfg["shot_capacity"]),
            )

            if positive_admitted:
                next_positive_entry_id += 1
                counters[
                    f"{item.origin}_positive_admission_count"
                ] += 1

            if removed_entry is not None:
                positive_replaced = True
                finalize_unknown_event(
                    event_rows,
                    run_name,
                    prefix_size,
                    unknown_count,
                    unknown_seed,
                    "positive",
                    removed_entry,
                    stream_step,
                    False,
                )

        lower_entropy = float(
            neg_cfg["entropy_threshold"]["lower"]
        )
        upper_entropy = float(
            neg_cfg["entropy_threshold"]["upper"]
        )
        negative_candidate = (
            bool(neg_cfg["enabled"])
            and lower_entropy
            < sample["normalized_entropy"]
            < upper_entropy
        )

        if negative_candidate:
            negative_entry = TrackedCacheEntry(
                entry_id=next_negative_entry_id,
                source_index=item.source_index,
                origin=item.origin,
                feature=sample["feature"].squeeze(0),
                entropy=sample["raw_entropy"],
                probability_map=sample["probability_map"],
                pseudo_label=sample["clip_prediction"],
                true_label=target,
                insertion_step=stream_step,
            )
            negative_admitted, removed_entry = update_cache(
                negative_cache,
                negative_entry,
                int(neg_cfg["shot_capacity"]),
            )

            if negative_admitted:
                next_negative_entry_id += 1
                counters[
                    f"{item.origin}_negative_admission_count"
                ] += 1

            if removed_entry is not None:
                negative_replaced = True
                finalize_unknown_event(
                    event_rows,
                    run_name,
                    prefix_size,
                    unknown_count,
                    unknown_seed,
                    "negative",
                    removed_entry,
                    stream_step,
                    False,
                )

        cache_logits = compute_origin_split_logits(
            sample["feature"],
            positive_cache,
            negative_cache,
            config,
            num_classes,
        )

        known_only_logits = (
            sample["clip_logits"]
            + cache_logits["positive_known"]
            - cache_logits["negative_known"]
        )
        final_logits = (
            known_only_logits
            + cache_logits["positive_unknown"]
            - cache_logits["negative_unknown"]
        )

        prediction_without_unknown_entries = int(
            known_only_logits.argmax(dim=1).item()
        )
        tda_prediction = int(final_logits.argmax(dim=1).item())
        clip_prediction = sample["clip_prediction"]

        tda_correct = (
            item.origin == "known"
            and tda_prediction == target
        )
        prediction_without_unknown_correct = (
            item.origin == "known"
            and prediction_without_unknown_entries == target
        )

        unknown_argmax_flip = (
            tda_prediction != prediction_without_unknown_entries
        )
        unknown_harmful_flip = (
            item.origin == "known"
            and prediction_without_unknown_correct
            and not tda_correct
        )
        unknown_helpful_flip = (
            item.origin == "known"
            and not prediction_without_unknown_correct
            and tda_correct
        )

        unknown_net_logits = (
            cache_logits["positive_unknown"]
            - cache_logits["negative_unknown"]
        )

        sample_rows.append(
            {
                "run_name": run_name,
                "prefix_size": prefix_size,
                "unknown_count": unknown_count,
                "unknown_seed": unknown_seed,
                "stream_step": stream_step,
                "origin": item.origin,
                "known_position": item.known_position,
                "source_index": item.source_index,
                "image_path": sample["image_path"],
                "true_label": target,
                "true_class": true_class,
                "clip_prediction": clip_prediction,
                "clip_prediction_class": classnames[clip_prediction],
                "tda_prediction": tda_prediction,
                "tda_prediction_class": classnames[tda_prediction],
                "prediction_without_unknown_entries": (
                    prediction_without_unknown_entries
                ),
                "prediction_without_unknown_entries_class": (
                    classnames[prediction_without_unknown_entries]
                ),
                "clip_correct": int(
                    item.origin == "known"
                    and clip_prediction == target
                ),
                "tda_correct": int(tda_correct),
                "prediction_without_unknown_entries_correct": int(
                    prediction_without_unknown_correct
                ),
                "normalized_entropy": sample["normalized_entropy"],
                "max_similarity": sample["max_similarity"],
                "probability_margin": sample["probability_margin"],
                "positive_admitted": int(positive_admitted),
                "negative_admitted": int(negative_admitted),
                "positive_replaced": int(positive_replaced),
                "negative_replaced": int(negative_replaced),
                "positive_cache_size": len(flatten_cache(positive_cache)),
                "negative_cache_size": len(flatten_cache(negative_cache)),
                "positive_unknown_cache_entries": cache_unknown_count(
                    positive_cache
                ),
                "negative_unknown_cache_entries": cache_unknown_count(
                    negative_cache
                ),
                "positive_contamination_ratio": (
                    cache_contamination_ratio(positive_cache)
                ),
                "negative_contamination_ratio": (
                    cache_contamination_ratio(negative_cache)
                ),
                "unknown_positive_logit_l1": float(
                    cache_logits["positive_unknown"].abs().sum().item()
                ),
                "unknown_negative_logit_l1": float(
                    cache_logits["negative_unknown"].abs().sum().item()
                ),
                "unknown_net_logit_l1": float(
                    unknown_net_logits.abs().sum().item()
                ),
                "unknown_net_logit_max_abs": float(
                    unknown_net_logits.abs().max().item()
                ),
                "unknown_argmax_flip": int(unknown_argmax_flip),
                "unknown_harmful_flip": int(unknown_harmful_flip),
                "unknown_helpful_flip": int(unknown_helpful_flip),
            }
        )

    final_step = len(stream)
    for cache_type, cache in [
        ("positive", positive_cache),
        ("negative", negative_cache),
    ]:
        for entry in flatten_cache(cache):
            finalize_unknown_event(
                event_rows,
                run_name,
                prefix_size,
                unknown_count,
                unknown_seed,
                cache_type,
                entry,
                final_step,
                True,
            )

    return SimulationResult(
        sample_df=pd.DataFrame(sample_rows),
        unknown_event_df=pd.DataFrame(event_rows),
        counters=counters,
    )


def build_paired_suffix(
    control_df: pd.DataFrame,
    contaminated_df: pd.DataFrame,
    prefix_size: int,
) -> pd.DataFrame:
    control_known = control_df[
        (control_df["origin"] == "known")
        & (control_df["known_position"] >= prefix_size)
    ].copy()
    contaminated_known = contaminated_df[
        (contaminated_df["origin"] == "known")
        & (contaminated_df["known_position"] >= prefix_size)
    ].copy()

    control_columns = [
        "known_position",
        "source_index",
        "true_label",
        "true_class",
        "clip_prediction",
        "clip_correct",
        "tda_prediction",
        "tda_prediction_class",
        "tda_correct",
    ]
    control_known = control_known[control_columns].rename(
        columns={
            "clip_prediction": "control_clip_prediction",
            "clip_correct": "control_clip_correct",
            "tda_prediction": "control_tda_prediction",
            "tda_prediction_class": "control_tda_prediction_class",
            "tda_correct": "control_tda_correct",
        }
    )

    contaminated_columns = [
        "known_position",
        "source_index",
        "stream_step",
        "tda_prediction",
        "tda_prediction_class",
        "tda_correct",
        "prediction_without_unknown_entries",
        "prediction_without_unknown_entries_class",
        "prediction_without_unknown_entries_correct",
        "positive_contamination_ratio",
        "negative_contamination_ratio",
        "positive_unknown_cache_entries",
        "negative_unknown_cache_entries",
        "unknown_positive_logit_l1",
        "unknown_negative_logit_l1",
        "unknown_net_logit_l1",
        "unknown_net_logit_max_abs",
        "unknown_argmax_flip",
        "unknown_harmful_flip",
        "unknown_helpful_flip",
    ]
    contaminated_known = contaminated_known[contaminated_columns].rename(
        columns={
            "tda_prediction": "contaminated_tda_prediction",
            "tda_prediction_class": (
                "contaminated_tda_prediction_class"
            ),
            "tda_correct": "contaminated_tda_correct",
        }
    )

    paired = control_known.merge(
        contaminated_known,
        on=["known_position", "source_index"],
        how="inner",
        validate="one_to_one",
    )

    paired["suffix_offset"] = (
        paired["known_position"].astype(int) - prefix_size
    )
    paired["paired_correctness_difference"] = (
        paired["contaminated_tda_correct"]
        - paired["control_tda_correct"]
    )
    paired["degraded"] = (
        (paired["control_tda_correct"] == 1)
        & (paired["contaminated_tda_correct"] == 0)
    ).astype(int)
    paired["improved"] = (
        (paired["control_tda_correct"] == 0)
        & (paired["contaminated_tda_correct"] == 1)
    ).astype(int)

    return paired


def summarize_run(
    paired_df: pd.DataFrame,
    contaminated_result: SimulationResult,
    event_df: pd.DataFrame,
    run_name: str,
    known_order_seed: int,
    prefix_size: int,
    unknown_count: int,
    unknown_seed: int,
) -> dict[str, Any]:
    contaminated_unknown = contaminated_result.sample_df[
        contaminated_result.sample_df["origin"] == "unknown"
    ]

    positive_lifetimes = event_df[
        event_df["cache_type"] == "positive"
    ]["lifetime"].tolist() if not event_df.empty else []
    negative_lifetimes = event_df[
        event_df["cache_type"] == "negative"
    ]["lifetime"].tolist() if not event_df.empty else []

    return {
        "run_name": run_name,
        "known_order_seed": known_order_seed,
        "prefix_size": prefix_size,
        "unknown_count": unknown_count,
        "unknown_seed": unknown_seed,
        "suffix_sample_count": len(paired_df),
        "control_suffix_accuracy": (
            100.0 * paired_df["control_tda_correct"].mean()
        ),
        "contaminated_suffix_accuracy": (
            100.0 * paired_df["contaminated_tda_correct"].mean()
        ),
        "paired_accuracy_difference_pp": (
            100.0 * paired_df["paired_correctness_difference"].mean()
        ),
        "degraded_known_count": int(paired_df["degraded"].sum()),
        "improved_known_count": int(paired_df["improved"].sum()),
        "unknown_positive_admission_count": (
            contaminated_result.counters[
                "unknown_positive_admission_count"
            ]
        ),
        "unknown_negative_admission_count": (
            contaminated_result.counters[
                "unknown_negative_admission_count"
            ]
        ),
        "unknown_positive_admission_rate": (
            contaminated_result.counters[
                "unknown_positive_admission_count"
            ]
            / unknown_count
            if unknown_count > 0
            else 0.0
        ),
        "unknown_negative_admission_rate": (
            contaminated_result.counters[
                "unknown_negative_admission_count"
            ]
            / unknown_count
            if unknown_count > 0
            else 0.0
        ),
        "peak_positive_contamination_ratio": float(
            contaminated_result.sample_df[
                "positive_contamination_ratio"
            ].max()
        ),
        "peak_negative_contamination_ratio": float(
            contaminated_result.sample_df[
                "negative_contamination_ratio"
            ].max()
        ),
        "mean_suffix_positive_contamination_ratio": float(
            paired_df["positive_contamination_ratio"].mean()
        ),
        "mean_suffix_negative_contamination_ratio": float(
            paired_df["negative_contamination_ratio"].mean()
        ),
        "unknown_argmax_flip_count": int(
            paired_df["unknown_argmax_flip"].sum()
        ),
        "unknown_harmful_flip_count": int(
            paired_df["unknown_harmful_flip"].sum()
        ),
        "unknown_helpful_flip_count": int(
            paired_df["unknown_helpful_flip"].sum()
        ),
        "mean_unknown_net_logit_l1": float(
            paired_df["unknown_net_logit_l1"].mean()
        ),
        "unknown_positive_lifetime_mean": (
            statistics.mean(positive_lifetimes)
            if positive_lifetimes
            else float("nan")
        ),
        "unknown_positive_lifetime_median": (
            statistics.median(positive_lifetimes)
            if positive_lifetimes
            else float("nan")
        ),
        "unknown_negative_lifetime_mean": (
            statistics.mean(negative_lifetimes)
            if negative_lifetimes
            else float("nan")
        ),
        "unknown_negative_lifetime_median": (
            statistics.median(negative_lifetimes)
            if negative_lifetimes
            else float("nan")
        ),
        "unknown_mean_entropy": (
            float(contaminated_unknown["normalized_entropy"].mean())
            if not contaminated_unknown.empty
            else float("nan")
        ),
        "unknown_mean_max_similarity": (
            float(contaminated_unknown["max_similarity"].mean())
            if not contaminated_unknown.empty
            else float("nan")
        ),
    }


def build_recovery_rows(
    paired_df: pd.DataFrame,
    run_name: str,
    prefix_size: int,
    unknown_count: int,
    unknown_seed: int,
    bin_size: int,
) -> list[dict[str, Any]]:
    working = paired_df.copy()
    working["recovery_bin"] = working["suffix_offset"] // bin_size
    rows: list[dict[str, Any]] = []

    for recovery_bin, bin_df in working.groupby(
        "recovery_bin",
        sort=True,
    ):
        end_offset = int(bin_df["suffix_offset"].max())
        cumulative = working[
            working["suffix_offset"] <= end_offset
        ]

        rows.append(
            {
                "run_name": run_name,
                "prefix_size": prefix_size,
                "unknown_count": unknown_count,
                "unknown_seed": unknown_seed,
                "recovery_bin": int(recovery_bin),
                "start_offset": int(bin_df["suffix_offset"].min()),
                "end_offset": end_offset,
                "num_samples": len(bin_df),
                "control_accuracy": (
                    100.0 * bin_df["control_tda_correct"].mean()
                ),
                "contaminated_accuracy": (
                    100.0 * bin_df["contaminated_tda_correct"].mean()
                ),
                "paired_accuracy_difference_pp": (
                    100.0
                    * bin_df["paired_correctness_difference"].mean()
                ),
                "cumulative_paired_accuracy_difference_pp": (
                    100.0
                    * cumulative[
                        "paired_correctness_difference"
                    ].mean()
                ),
                "degraded_count": int(bin_df["degraded"].sum()),
                "improved_count": int(bin_df["improved"].sum()),
                "mean_positive_contamination_ratio": float(
                    bin_df["positive_contamination_ratio"].mean()
                ),
                "end_positive_contamination_ratio": float(
                    bin_df.iloc[-1]["positive_contamination_ratio"]
                ),
                "mean_negative_contamination_ratio": float(
                    bin_df["negative_contamination_ratio"].mean()
                ),
                "end_negative_contamination_ratio": float(
                    bin_df.iloc[-1]["negative_contamination_ratio"]
                ),
                "unknown_argmax_flip_count": int(
                    bin_df["unknown_argmax_flip"].sum()
                ),
                "unknown_harmful_flip_count": int(
                    bin_df["unknown_harmful_flip"].sum()
                ),
            }
        )

    return rows


def build_class_rows(
    paired_df: pd.DataFrame,
    run_name: str,
    prefix_size: int,
    unknown_count: int,
    unknown_seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for true_class, class_df in paired_df.groupby(
        "true_class",
        sort=True,
    ):
        rows.append(
            {
                "run_name": run_name,
                "prefix_size": prefix_size,
                "unknown_count": unknown_count,
                "unknown_seed": unknown_seed,
                "true_class": true_class,
                "num_samples": len(class_df),
                "control_accuracy": (
                    100.0 * class_df["control_tda_correct"].mean()
                ),
                "contaminated_accuracy": (
                    100.0
                    * class_df["contaminated_tda_correct"].mean()
                ),
                "accuracy_difference_pp": (
                    100.0
                    * class_df[
                        "paired_correctness_difference"
                    ].mean()
                ),
                "degraded_count": int(class_df["degraded"].sum()),
                "improved_count": int(class_df["improved"].sum()),
            }
        )

    return rows


def save_plots(
    summary_df: pd.DataFrame,
    recovery_df: pd.DataFrame,
    figure_dir: Path,
    main_prefix_size: int,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)

    aggregated = (
        summary_df.groupby(
            ["prefix_size", "unknown_count"],
            as_index=False,
        )
        .agg(
            mean_difference=(
                "paired_accuracy_difference_pp",
                "mean",
            ),
            std_difference=(
                "paired_accuracy_difference_pp",
                "std",
            ),
            mean_positive_admission_rate=(
                "unknown_positive_admission_rate",
                "mean",
            ),
            mean_negative_admission_rate=(
                "unknown_negative_admission_rate",
                "mean",
            ),
        )
        .fillna(0.0)
    )

    plt.figure(figsize=(9, 6))
    for prefix_size, prefix_df in aggregated.groupby(
        "prefix_size",
        sort=True,
    ):
        plt.errorbar(
            prefix_df["unknown_count"],
            prefix_df["mean_difference"],
            yerr=prefix_df["std_difference"],
            marker="o",
            capsize=4,
            label=f"Known prefix = {prefix_size}",
        )
    plt.axhline(0.0, linewidth=1.0)
    plt.xlabel("Number of inserted unknown samples")
    plt.ylabel("Paired suffix accuracy difference (percentage points)")
    plt.title("Known-suffix accuracy under unknown-class contamination")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        figure_dir / "unknown_contamination_paired_accuracy.png",
        dpi=200,
    )
    plt.close()

    main_recovery = recovery_df[
        recovery_df["prefix_size"] == main_prefix_size
    ].copy()

    recovery_aggregated = (
        main_recovery.groupby(
            ["unknown_count", "recovery_bin", "end_offset"],
            as_index=False,
        )
        .agg(
            mean_difference=(
                "paired_accuracy_difference_pp",
                "mean",
            ),
            mean_positive_contamination=(
                "mean_positive_contamination_ratio",
                "mean",
            ),
        )
    )

    plt.figure(figsize=(10, 6))
    for unknown_count, count_df in recovery_aggregated.groupby(
        "unknown_count",
        sort=True,
    ):
        if unknown_count == 0:
            continue
        plt.plot(
            count_df["end_offset"],
            count_df["mean_difference"],
            marker="o",
            markersize=3,
            label=f"Unknown count = {unknown_count}",
        )
    plt.axhline(0.0, linewidth=1.0)
    plt.xlabel("Known samples after the unknown block")
    plt.ylabel("Paired accuracy difference (percentage points)")
    plt.title(
        f"Accuracy recovery after contamination (prefix = {main_prefix_size})"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        figure_dir / "unknown_contamination_recovery_curve.png",
        dpi=200,
    )
    plt.close()

    plt.figure(figsize=(10, 6))
    for unknown_count, count_df in recovery_aggregated.groupby(
        "unknown_count",
        sort=True,
    ):
        if unknown_count == 0:
            continue
        plt.plot(
            count_df["end_offset"],
            count_df["mean_positive_contamination"],
            marker="o",
            markersize=3,
            label=f"Unknown count = {unknown_count}",
        )
    plt.xlabel("Known samples after the unknown block")
    plt.ylabel("Positive cache contamination ratio")
    plt.title(
        f"Positive cache recovery after contamination (prefix = {main_prefix_size})"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        figure_dir / "unknown_contamination_cache_recovery.png",
        dpi=200,
    )
    plt.close()

    plt.figure(figsize=(9, 6))
    for prefix_size, prefix_df in aggregated.groupby(
        "prefix_size",
        sort=True,
    ):
        plt.plot(
            prefix_df["unknown_count"],
            prefix_df["mean_positive_admission_rate"],
            marker="o",
            label=f"Positive, prefix = {prefix_size}",
        )
        plt.plot(
            prefix_df["unknown_count"],
            prefix_df["mean_negative_admission_rate"],
            marker="s",
            linestyle="--",
            label=f"Negative, prefix = {prefix_size}",
        )
    plt.xlabel("Number of inserted unknown samples")
    plt.ylabel("Unknown admission rate")
    plt.title("Unknown sample admission into TDA caches")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(
        figure_dir / "unknown_contamination_admission_rate.png",
        dpi=200,
    )
    plt.close()


def add_control_rows(
    prefix_size: int,
    known_order_seed: int,
    control_df: pd.DataFrame,
    bin_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    suffix = control_df[
        (control_df["origin"] == "known")
        & (control_df["known_position"] >= prefix_size)
    ].copy()
    suffix["suffix_offset"] = (
        suffix["known_position"].astype(int) - prefix_size
    )

    summary = {
        "run_name": f"prefix_{prefix_size}_unknown_0_control",
        "known_order_seed": known_order_seed,
        "prefix_size": prefix_size,
        "unknown_count": 0,
        "unknown_seed": -1,
        "suffix_sample_count": len(suffix),
        "control_suffix_accuracy": 100.0 * suffix["tda_correct"].mean(),
        "contaminated_suffix_accuracy": 100.0 * suffix["tda_correct"].mean(),
        "paired_accuracy_difference_pp": 0.0,
        "degraded_known_count": 0,
        "improved_known_count": 0,
        "unknown_positive_admission_count": 0,
        "unknown_negative_admission_count": 0,
        "unknown_positive_admission_rate": 0.0,
        "unknown_negative_admission_rate": 0.0,
        "peak_positive_contamination_ratio": 0.0,
        "peak_negative_contamination_ratio": 0.0,
        "mean_suffix_positive_contamination_ratio": 0.0,
        "mean_suffix_negative_contamination_ratio": 0.0,
        "unknown_argmax_flip_count": 0,
        "unknown_harmful_flip_count": 0,
        "unknown_helpful_flip_count": 0,
        "mean_unknown_net_logit_l1": 0.0,
        "unknown_positive_lifetime_mean": float("nan"),
        "unknown_positive_lifetime_median": float("nan"),
        "unknown_negative_lifetime_mean": float("nan"),
        "unknown_negative_lifetime_median": float("nan"),
        "unknown_mean_entropy": float("nan"),
        "unknown_mean_max_similarity": float("nan"),
    }

    recovery_rows: list[dict[str, Any]] = []
    suffix["recovery_bin"] = suffix["suffix_offset"] // bin_size
    for recovery_bin, bin_df in suffix.groupby(
        "recovery_bin",
        sort=True,
    ):
        recovery_rows.append(
            {
                "run_name": summary["run_name"],
                "prefix_size": prefix_size,
                "unknown_count": 0,
                "unknown_seed": -1,
                "recovery_bin": int(recovery_bin),
                "start_offset": int(bin_df["suffix_offset"].min()),
                "end_offset": int(bin_df["suffix_offset"].max()),
                "num_samples": len(bin_df),
                "control_accuracy": 100.0 * bin_df["tda_correct"].mean(),
                "contaminated_accuracy": 100.0 * bin_df["tda_correct"].mean(),
                "paired_accuracy_difference_pp": 0.0,
                "cumulative_paired_accuracy_difference_pp": 0.0,
                "degraded_count": 0,
                "improved_count": 0,
                "mean_positive_contamination_ratio": 0.0,
                "end_positive_contamination_ratio": 0.0,
                "mean_negative_contamination_ratio": 0.0,
                "end_negative_contamination_ratio": 0.0,
                "unknown_argmax_flip_count": 0,
                "unknown_harmful_flip_count": 0,
            }
        )

    class_rows: list[dict[str, Any]] = []
    for true_class, class_df in suffix.groupby("true_class", sort=True):
        class_rows.append(
            {
                "run_name": summary["run_name"],
                "prefix_size": prefix_size,
                "unknown_count": 0,
                "unknown_seed": -1,
                "true_class": true_class,
                "num_samples": len(class_df),
                "control_accuracy": 100.0 * class_df["tda_correct"].mean(),
                "contaminated_accuracy": 100.0 * class_df["tda_correct"].mean(),
                "accuracy_difference_pp": 0.0,
                "degraded_count": 0,
                "improved_count": 0,
            }
        )

    return summary, recovery_rows, class_rows


def main() -> None:
    args = parse_arguments()

    known_feature_path = Path(args.known_features).expanduser().resolve()
    unknown_feature_path = Path(args.unknown_features).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    for required_path in [
        known_feature_path,
        unknown_feature_path,
        config_path,
    ]:
        if not required_path.is_file():
            raise FileNotFoundError(
                f"Required file was not found: {required_path}"
            )

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Select a GPU runtime or use --device cpu."
        )

    device = torch.device(args.device)
    known_artifact = load_torch_artifact(known_feature_path)
    unknown_artifact = load_torch_artifact(unknown_feature_path)
    validate_artifacts(known_artifact, unknown_artifact)
    config = load_config(config_path)

    known_sample_count = len(known_artifact["labels"])
    unknown_sample_count = len(unknown_artifact["features"])

    prefix_sizes = sorted(set(args.prefix_sizes))
    unknown_counts = sorted(set(args.unknown_counts))
    unknown_seeds = sorted(set(args.unknown_seeds))

    if 0 not in unknown_counts:
        unknown_counts = [0] + unknown_counts

    for prefix_size in prefix_sizes:
        if not 0 <= prefix_size < known_sample_count:
            raise ValueError(
                f"Invalid prefix size {prefix_size} for "
                f"{known_sample_count} known samples."
            )

    for unknown_count in unknown_counts:
        if unknown_count < 0 or unknown_count > unknown_sample_count:
            raise ValueError(
                f"Invalid unknown count {unknown_count} for "
                f"{unknown_sample_count} available unknown samples."
            )

    raw_dir = output_root / "raw" / "unknown_contamination"
    summary_dir = output_root / "summaries"
    figure_dir = output_root / "figures"
    raw_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    known_order = build_known_order(
        known_sample_count,
        args.known_order_seed,
    )
    control_stream = build_control_stream(known_order)

    print("Device:", device)
    print("Known sample count:", known_sample_count)
    print("Available unknown sample count:", unknown_sample_count)
    print("Known order seed:", args.known_order_seed)
    print("Prefix sizes:", prefix_sizes)
    print("Unknown counts:", unknown_counts)
    print("Unknown seeds:", unknown_seeds)

    control_result = simulate_stream(
        known_artifact,
        unknown_artifact,
        config,
        control_stream,
        run_name="control_known_only",
        prefix_size=0,
        unknown_count=0,
        unknown_seed=-1,
        device=device,
    )

    summary_rows: list[dict[str, Any]] = []
    recovery_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    paired_frames: list[pd.DataFrame] = []
    sample_frames: list[pd.DataFrame] = []
    event_frames: list[pd.DataFrame] = []

    for prefix_size in prefix_sizes:
        control_summary, control_recovery, control_classes = add_control_rows(
            prefix_size,
            args.known_order_seed,
            control_result.sample_df,
            args.recovery_bin_size,
        )
        summary_rows.append(control_summary)
        recovery_rows.extend(control_recovery)
        class_rows.extend(control_classes)

        for unknown_count in unknown_counts:
            if unknown_count == 0:
                continue

            for unknown_seed in unknown_seeds:
                run_name = (
                    f"prefix_{prefix_size}_unknown_{unknown_count}_"
                    f"seed_{unknown_seed}"
                )
                print("Running:", run_name)

                unknown_indices = select_unknown_indices(
                    unknown_sample_count,
                    unknown_count,
                    unknown_seed,
                )
                contaminated_stream = build_contaminated_stream(
                    known_order,
                    prefix_size,
                    unknown_indices,
                )
                contaminated_result = simulate_stream(
                    known_artifact,
                    unknown_artifact,
                    config,
                    contaminated_stream,
                    run_name=run_name,
                    prefix_size=prefix_size,
                    unknown_count=unknown_count,
                    unknown_seed=unknown_seed,
                    device=device,
                )

                paired_df = build_paired_suffix(
                    control_result.sample_df,
                    contaminated_result.sample_df,
                    prefix_size,
                )
                paired_df.insert(0, "run_name", run_name)
                paired_df.insert(1, "prefix_size", prefix_size)
                paired_df.insert(2, "unknown_count", unknown_count)
                paired_df.insert(3, "unknown_seed", unknown_seed)
                paired_frames.append(paired_df)

                sample_frames.append(contaminated_result.sample_df)
                if not contaminated_result.unknown_event_df.empty:
                    event_frames.append(
                        contaminated_result.unknown_event_df
                    )

                summary_rows.append(
                    summarize_run(
                        paired_df,
                        contaminated_result,
                        contaminated_result.unknown_event_df,
                        run_name,
                        args.known_order_seed,
                        prefix_size,
                        unknown_count,
                        unknown_seed,
                    )
                )
                recovery_rows.extend(
                    build_recovery_rows(
                        paired_df,
                        run_name,
                        prefix_size,
                        unknown_count,
                        unknown_seed,
                        args.recovery_bin_size,
                    )
                )
                class_rows.extend(
                    build_class_rows(
                        paired_df,
                        run_name,
                        prefix_size,
                        unknown_count,
                        unknown_seed,
                    )
                )

    summary_df = pd.DataFrame(summary_rows)
    recovery_df = pd.DataFrame(recovery_rows)
    class_df = pd.DataFrame(class_rows)
    paired_all_df = (
        pd.concat(paired_frames, ignore_index=True)
        if paired_frames
        else pd.DataFrame()
    )
    samples_all_df = (
        pd.concat(sample_frames, ignore_index=True)
        if sample_frames
        else pd.DataFrame()
    )
    events_all_df = (
        pd.concat(event_frames, ignore_index=True)
        if event_frames
        else pd.DataFrame()
    )

    summary_path = summary_dir / "unknown_contamination_summary.csv"
    recovery_path = summary_dir / "unknown_contamination_recovery.csv"
    class_path = summary_dir / "unknown_contamination_class_effects.csv"
    paired_path = raw_dir / "unknown_contamination_paired_suffix.csv"
    samples_path = raw_dir / "unknown_contamination_all_samples.csv"
    events_path = raw_dir / "unknown_contamination_cache_events.csv"

    summary_df.to_csv(summary_path, index=False)
    recovery_df.to_csv(recovery_path, index=False)
    class_df.to_csv(class_path, index=False)
    paired_all_df.to_csv(paired_path, index=False)
    samples_all_df.to_csv(samples_path, index=False)
    events_all_df.to_csv(events_path, index=False)

    metadata = {
        "known_feature_path": str(known_feature_path),
        "unknown_feature_path": str(unknown_feature_path),
        "config_path": str(config_path),
        "known_order_seed": args.known_order_seed,
        "prefix_sizes": prefix_sizes,
        "unknown_counts": unknown_counts,
        "unknown_seeds": unknown_seeds,
        "recovery_bin_size": args.recovery_bin_size,
        "known_sample_count": known_sample_count,
        "unknown_sample_count": unknown_sample_count,
        "unknown_source_name": unknown_artifact.get(
            "source_name",
            "unknown",
        ),
        "direct_contribution_note": (
            "Predictions without unknown entries remove currently active "
            "unknown-origin cache entries but do not restore known entries "
            "that may have been displaced earlier."
        ),
    }
    metadata_path = summary_dir / "unknown_contamination_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    save_plots(
        summary_df,
        recovery_df,
        figure_dir,
        args.main_prefix_size,
    )

    display_columns = [
        "run_name",
        "control_suffix_accuracy",
        "contaminated_suffix_accuracy",
        "paired_accuracy_difference_pp",
        "unknown_positive_admission_rate",
        "unknown_negative_admission_rate",
        "peak_positive_contamination_ratio",
        "unknown_harmful_flip_count",
    ]
    print()
    print(summary_df[display_columns].to_string(index=False))
    print()
    print("Saved summary:", summary_path)
    print("Saved recovery metrics:", recovery_path)
    print("Saved class effects:", class_path)
    print("Saved paired sample results:", paired_path)
    print("Saved all sample results:", samples_path)
    print("Saved cache events:", events_path)
    print("Saved figures:", figure_dir)


if __name__ == "__main__":
    main()
