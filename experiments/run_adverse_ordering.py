from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
import yaml


@dataclass
class CacheEntry:
    entry_id: int
    source_index: int
    feature: torch.Tensor
    entropy: float
    probability_map: torch.Tensor | None
    pseudo_label: int
    true_label: int
    insertion_step: int

    @property
    def is_correct(self) -> bool:
        return self.pseudo_label == self.true_label


@dataclass
class ExperimentRun:
    name: str
    condition: str
    seed: int
    order: list[int]
    stress_prefix_size: int = 0


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate TDA under adverse test ordering conditions."
    )
    parser.add_argument("--features", required=True)
    parser.add_argument("--config", default="./configs/caltech101.yaml")
    parser.add_argument("--output-root", default="./results")
    parser.add_argument("--interval-size", type=int, default=100)
    parser.add_argument("--random-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--deterministic-seed", type=int, default=0)
    parser.add_argument(
        "--stress-prefix-size",
        type=int,
        default=0,
        help="Number of confidently wrong samples placed first. Zero uses all confidently wrong samples.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=[
            "random",
            "class_balanced_easy_first",
            "class_balanced_uncertain_first",
            "confidently_wrong_first",
        ],
        choices=[
            "random",
            "class_balanced_easy_first",
            "class_balanced_uncertain_first",
            "confidently_wrong_first",
        ],
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--wandb-log", action="store_true")
    parser.add_argument("--wandb-project", default="TDA-VisualMedia")
    return parser.parse_args()


def load_torch_artifact(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def round_robin_from_groups(groups: dict[int, list[int]]) -> list[int]:
    queues = {
        key: deque(values)
        for key, values in sorted(groups.items(), key=lambda item: item[0])
    }
    output: list[int] = []

    while any(queues.values()):
        for key in sorted(queues):
            if queues[key]:
                output.append(queues[key].popleft())

    return output


def build_class_balanced_order(
    labels: torch.Tensor,
    normalized_entropy: torch.Tensor,
    descending: bool,
) -> list[int]:
    groups: dict[int, list[int]] = defaultdict(list)

    for index, label in enumerate(labels.tolist()):
        groups[int(label)].append(index)

    for label, indices in groups.items():
        groups[label] = sorted(
            indices,
            key=lambda index: (
                float(normalized_entropy[index]),
                index,
            ),
            reverse=descending,
        )

    return round_robin_from_groups(groups)


def build_confidently_wrong_first_order(
    labels: torch.Tensor,
    predictions: torch.Tensor,
    normalized_entropy: torch.Tensor,
    seed: int,
    prefix_size: int,
) -> tuple[list[int], int]:
    wrong_groups: dict[int, list[int]] = defaultdict(list)

    for index, (label, prediction) in enumerate(
        zip(labels.tolist(), predictions.tolist())
    ):
        if int(label) != int(prediction):
            wrong_groups[int(prediction)].append(index)

    for predicted_class, indices in wrong_groups.items():
        wrong_groups[predicted_class] = sorted(
            indices,
            key=lambda index: (
                float(normalized_entropy[index]),
                index,
            ),
        )

    confidently_wrong = round_robin_from_groups(wrong_groups)

    if prefix_size > 0:
        confidently_wrong = confidently_wrong[:prefix_size]

    prefix_set = set(confidently_wrong)
    remaining = [
        index
        for index in range(len(labels))
        if index not in prefix_set
    ]

    random.Random(seed).shuffle(remaining)
    return confidently_wrong + remaining, len(confidently_wrong)


def build_runs(
    artifact: dict[str, Any],
    conditions: Iterable[str],
    random_seeds: list[int],
    deterministic_seed: int,
    stress_prefix_size: int,
) -> list[ExperimentRun]:
    labels = artifact["labels"]
    predictions = artifact["clip_predictions"]
    normalized_entropy = artifact["normalized_entropy"]
    sample_count = len(labels)
    runs: list[ExperimentRun] = []

    if "random" in conditions:
        for seed in random_seeds:
            order = list(range(sample_count))
            random.Random(seed).shuffle(order)
            runs.append(
                ExperimentRun(
                    name=f"random_seed_{seed}",
                    condition="random",
                    seed=seed,
                    order=order,
                )
            )

    if "class_balanced_easy_first" in conditions:
        runs.append(
            ExperimentRun(
                name="class_balanced_easy_first",
                condition="class_balanced_easy_first",
                seed=deterministic_seed,
                order=build_class_balanced_order(
                    labels,
                    normalized_entropy,
                    descending=False,
                ),
            )
        )

    if "class_balanced_uncertain_first" in conditions:
        runs.append(
            ExperimentRun(
                name="class_balanced_uncertain_first",
                condition="class_balanced_uncertain_first",
                seed=deterministic_seed,
                order=build_class_balanced_order(
                    labels,
                    normalized_entropy,
                    descending=True,
                ),
            )
        )

    if "confidently_wrong_first" in conditions:
        order, actual_prefix_size = build_confidently_wrong_first_order(
            labels,
            predictions,
            normalized_entropy,
            deterministic_seed,
            stress_prefix_size,
        )
        runs.append(
            ExperimentRun(
                name="confidently_wrong_first",
                condition="confidently_wrong_first",
                seed=deterministic_seed,
                order=order,
                stress_prefix_size=actual_prefix_size,
            )
        )

    return runs


def flatten_cache(cache: dict[int, list[CacheEntry]]) -> list[CacheEntry]:
    entries: list[CacheEntry] = []
    for class_index in sorted(cache):
        entries.extend(cache[class_index])
    return entries


def update_cache(
    cache: dict[int, list[CacheEntry]],
    entry: CacheEntry,
    shot_capacity: int,
) -> tuple[bool, CacheEntry | None]:
    class_entries = cache.setdefault(entry.pseudo_label, [])
    removed_entry: CacheEntry | None = None
    admitted = False

    if len(class_entries) < shot_capacity:
        class_entries.append(entry)
        admitted = True
    elif entry.entropy < class_entries[-1].entropy:
        removed_entry = class_entries[-1]
        class_entries[-1] = entry
        admitted = True

    class_entries.sort(key=lambda cache_entry: cache_entry.entropy)
    return admitted, removed_entry


def compute_positive_cache_logits(
    query_feature: torch.Tensor,
    cache: dict[int, list[CacheEntry]],
    alpha: float,
    beta: float,
    num_classes: int,
) -> torch.Tensor:
    entries = flatten_cache(cache)
    if not entries:
        return torch.zeros(
            (1, num_classes),
            device=query_feature.device,
            dtype=query_feature.dtype,
        )

    keys = torch.stack([entry.feature for entry in entries], dim=0)
    labels = torch.tensor(
        [entry.pseudo_label for entry in entries],
        device=query_feature.device,
        dtype=torch.long,
    )
    values = F.one_hot(labels, num_classes=num_classes).to(query_feature.dtype)
    affinity = query_feature @ keys.T
    weights = torch.exp(-beta * (1.0 - affinity))
    return alpha * (weights @ values)


def compute_negative_cache_logits(
    query_feature: torch.Tensor,
    cache: dict[int, list[CacheEntry]],
    alpha: float,
    beta: float,
    num_classes: int,
    mask_lower: float,
    mask_upper: float,
) -> torch.Tensor:
    entries = flatten_cache(cache)
    if not entries:
        return torch.zeros(
            (1, num_classes),
            device=query_feature.device,
            dtype=query_feature.dtype,
        )

    keys = torch.stack([entry.feature for entry in entries], dim=0)
    probability_maps = torch.stack(
        [entry.probability_map for entry in entries if entry.probability_map is not None],
        dim=0,
    )
    masks = (
        (probability_maps > mask_lower)
        & (probability_maps < mask_upper)
    ).to(query_feature.dtype)
    affinity = query_feature @ keys.T
    weights = torch.exp(-beta * (1.0 - affinity))
    return alpha * (weights @ masks)


def cache_purity(cache: dict[int, list[CacheEntry]]) -> float:
    entries = flatten_cache(cache)
    if not entries:
        return float("nan")
    return sum(entry.is_correct for entry in entries) / len(entries)


def finalize_removed_entry(
    event_rows: list[dict[str, Any]],
    run: ExperimentRun,
    entry: CacheEntry,
    removal_step: int,
    censored: bool,
) -> None:
    event_rows.append(
        {
            "run_name": run.name,
            "condition": run.condition,
            "seed": run.seed,
            "entry_id": entry.entry_id,
            "source_index": entry.source_index,
            "pseudo_label": entry.pseudo_label,
            "true_label": entry.true_label,
            "is_correct": int(entry.is_correct),
            "insertion_step": entry.insertion_step,
            "removal_step": removal_step,
            "lifetime": removal_step - entry.insertion_step,
            "censored": int(censored),
        }
    )


def simulate_run(
    artifact: dict[str, Any],
    config: dict[str, Any],
    run: ExperimentRun,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    features = artifact["features"]
    clip_logits_all = artifact["clip_logits"]
    probabilities = artifact["probabilities"]
    raw_entropy = artifact["raw_entropy"]
    normalized_entropy = artifact["normalized_entropy"]
    clip_predictions = artifact["clip_predictions"]
    labels = artifact["labels"]
    original_indices = artifact["original_indices"]
    image_paths = artifact["image_paths"]
    classnames = artifact["classnames"]
    num_classes = int(artifact["num_classes"])

    pos_cfg = config["positive"]
    neg_cfg = config["negative"]

    positive_cache: dict[int, list[CacheEntry]] = {}
    negative_cache: dict[int, list[CacheEntry]] = {}
    sample_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []

    next_positive_entry_id = 0
    next_negative_entry_id = 0
    cumulative_clip_correct = 0
    cumulative_tda_correct = 0
    cumulative_clip_correct_tda_wrong = 0
    positive_admission_count = 0
    positive_correct_admission_count = 0

    for step, source_index in enumerate(run.order):
        feature = features[source_index].to(device=device).unsqueeze(0)
        clip_logits = clip_logits_all[source_index].to(device=device).unsqueeze(0)
        probability_map = probabilities[source_index].to(device=device)
        target = int(labels[source_index])
        clip_prediction = int(clip_predictions[source_index])
        sample_raw_entropy = float(raw_entropy[source_index])
        sample_normalized_entropy = float(normalized_entropy[source_index])

        positive_admitted = False
        positive_replaced = False
        negative_admitted = False
        negative_replaced = False

        if bool(pos_cfg["enabled"]):
            positive_entry = CacheEntry(
                entry_id=next_positive_entry_id,
                source_index=source_index,
                feature=feature.squeeze(0),
                entropy=sample_raw_entropy,
                probability_map=None,
                pseudo_label=clip_prediction,
                true_label=target,
                insertion_step=step,
            )
            positive_admitted, removed_entry = update_cache(
                positive_cache,
                positive_entry,
                int(pos_cfg["shot_capacity"]),
            )

            if positive_admitted:
                next_positive_entry_id += 1
                positive_admission_count += 1
                positive_correct_admission_count += int(positive_entry.is_correct)

            if removed_entry is not None:
                positive_replaced = True
                finalize_removed_entry(
                    event_rows,
                    run,
                    removed_entry,
                    removal_step=step,
                    censored=False,
                )

        lower_entropy = float(neg_cfg["entropy_threshold"]["lower"])
        upper_entropy = float(neg_cfg["entropy_threshold"]["upper"])
        negative_candidate = (
            bool(neg_cfg["enabled"])
            and lower_entropy < sample_normalized_entropy < upper_entropy
        )

        if negative_candidate:
            negative_entry = CacheEntry(
                entry_id=next_negative_entry_id,
                source_index=source_index,
                feature=feature.squeeze(0),
                entropy=sample_raw_entropy,
                probability_map=probability_map,
                pseudo_label=clip_prediction,
                true_label=target,
                insertion_step=step,
            )
            negative_admitted, removed_negative_entry = update_cache(
                negative_cache,
                negative_entry,
                int(neg_cfg["shot_capacity"]),
            )
            if negative_admitted:
                next_negative_entry_id += 1
            negative_replaced = removed_negative_entry is not None

        positive_logits = torch.zeros_like(clip_logits)
        negative_logits = torch.zeros_like(clip_logits)
        final_logits = clip_logits.clone()

        if bool(pos_cfg["enabled"]) and positive_cache:
            positive_logits = compute_positive_cache_logits(
                feature,
                positive_cache,
                float(pos_cfg["alpha"]),
                float(pos_cfg["beta"]),
                num_classes,
            )
            final_logits = final_logits + positive_logits

        if bool(neg_cfg["enabled"]) and negative_cache:
            negative_logits = compute_negative_cache_logits(
                feature,
                negative_cache,
                float(neg_cfg["alpha"]),
                float(neg_cfg["beta"]),
                num_classes,
                float(neg_cfg["mask_threshold"]["lower"]),
                float(neg_cfg["mask_threshold"]["upper"]),
            )
            final_logits = final_logits - negative_logits

        tda_prediction = int(final_logits.argmax(dim=1).item())
        clip_correct = clip_prediction == target
        tda_correct = tda_prediction == target
        clip_correct_tda_wrong = clip_correct and not tda_correct

        cumulative_clip_correct += int(clip_correct)
        cumulative_tda_correct += int(tda_correct)
        cumulative_clip_correct_tda_wrong += int(clip_correct_tda_wrong)

        current_positive_entries = flatten_cache(positive_cache)
        current_wrong_entries = sum(not entry.is_correct for entry in current_positive_entries)
        current_purity = cache_purity(positive_cache)

        sample_rows.append(
            {
                "run_name": run.name,
                "condition": run.condition,
                "seed": run.seed,
                "step": step,
                "source_index": source_index,
                "original_index": int(original_indices[source_index]),
                "image_path": image_paths[source_index],
                "true_label": target,
                "true_class": classnames[target],
                "clip_prediction": clip_prediction,
                "clip_prediction_class": classnames[clip_prediction],
                "tda_prediction": tda_prediction,
                "tda_prediction_class": classnames[tda_prediction],
                "clip_correct": int(clip_correct),
                "tda_correct": int(tda_correct),
                "clip_correct_tda_wrong": int(clip_correct_tda_wrong),
                "raw_entropy": sample_raw_entropy,
                "normalized_entropy": sample_normalized_entropy,
                "max_similarity": float(artifact["max_similarity"][source_index]),
                "probability_margin": float(artifact["probability_margin"][source_index]),
                "clip_wrong": int(not clip_correct),
                "is_stress_prefix": int(
                    run.condition == "confidently_wrong_first"
                    and step < run.stress_prefix_size
                ),
                "positive_admitted": int(positive_admitted),
                "positive_replaced": int(positive_replaced),
                "negative_admitted": int(negative_admitted),
                "negative_replaced": int(negative_replaced),
                "positive_cache_size": len(current_positive_entries),
                "negative_cache_size": len(flatten_cache(negative_cache)),
                "positive_cache_purity": current_purity,
                "wrong_positive_cache_entries": current_wrong_entries,
                "positive_cache_argmax": int(positive_logits.argmax(dim=1).item()),
                "negative_cache_argmax": int(negative_logits.argmax(dim=1).item()),
                "positive_logit_max": float(positive_logits.max().item()),
                "negative_logit_max": float(negative_logits.max().item()),
                "cumulative_clip_accuracy": 100.0 * cumulative_clip_correct / (step + 1),
                "cumulative_tda_accuracy": 100.0 * cumulative_tda_correct / (step + 1),
                "cumulative_tda_minus_clip_pp": (
                    100.0 * (cumulative_tda_correct - cumulative_clip_correct) / (step + 1)
                ),
                "cumulative_clip_correct_tda_wrong": cumulative_clip_correct_tda_wrong,
            }
        )

    final_step = len(run.order)
    for active_entry in flatten_cache(positive_cache):
        finalize_removed_entry(
            event_rows,
            run,
            active_entry,
            removal_step=final_step,
            censored=True,
        )

    sample_df = pd.DataFrame(sample_rows)
    event_df = pd.DataFrame(event_rows)

    wrong_events = event_df[event_df["is_correct"] == 0] if not event_df.empty else event_df
    wrong_lifetimes = wrong_events["lifetime"].tolist() if not wrong_events.empty else []

    summary = {
        "run_name": run.name,
        "condition": run.condition,
        "seed": run.seed,
        "num_samples": len(sample_df),
        "stress_prefix_size": run.stress_prefix_size,
        "clip_accuracy": 100.0 * sample_df["clip_correct"].mean(),
        "tda_accuracy": 100.0 * sample_df["tda_correct"].mean(),
        "tda_minus_clip_pp": 100.0 * (
            sample_df["tda_correct"].mean() - sample_df["clip_correct"].mean()
        ),
        "clip_correct_tda_wrong_count": int(sample_df["clip_correct_tda_wrong"].sum()),
        "positive_admission_count": positive_admission_count,
        "positive_admission_purity": (
            positive_correct_admission_count / positive_admission_count
            if positive_admission_count > 0
            else float("nan")
        ),
        "final_positive_cache_purity": float(sample_df.iloc[-1]["positive_cache_purity"]),
        "mean_positive_cache_purity": float(sample_df["positive_cache_purity"].mean()),
        "wrong_cache_entry_count": len(wrong_lifetimes),
        "wrong_cache_lifetime_mean": (
            statistics.mean(wrong_lifetimes) if wrong_lifetimes else float("nan")
        ),
        "wrong_cache_lifetime_median": (
            statistics.median(wrong_lifetimes) if wrong_lifetimes else float("nan")
        ),
        "wrong_cache_lifetime_max": max(wrong_lifetimes) if wrong_lifetimes else float("nan"),
        "wrong_cache_survival_to_end_rate": (
            float(wrong_events["censored"].mean()) if not wrong_events.empty else float("nan")
        ),
    }

    return sample_df, event_df, summary


def build_interval_metrics(sample_df: pd.DataFrame, interval_size: int) -> pd.DataFrame:
    working_df = sample_df.copy()
    working_df["interval_id"] = working_df["step"] // interval_size
    rows: list[dict[str, Any]] = []

    for interval_id, interval_df in working_df.groupby("interval_id", sort=True):
        end_row = interval_df.iloc[-1]
        rows.append(
            {
                "run_name": end_row["run_name"],
                "condition": end_row["condition"],
                "seed": int(end_row["seed"]),
                "interval_id": int(interval_id),
                "start_step": int(interval_df["step"].min()),
                "end_step": int(interval_df["step"].max()),
                "num_samples": len(interval_df),
                "clip_interval_accuracy": 100.0 * interval_df["clip_correct"].mean(),
                "tda_interval_accuracy": 100.0 * interval_df["tda_correct"].mean(),
                "interval_tda_minus_clip_pp": 100.0 * (
                    interval_df["tda_correct"].mean()
                    - interval_df["clip_correct"].mean()
                ),
                "clip_correct_tda_wrong_interval": int(
                    interval_df["clip_correct_tda_wrong"].sum()
                ),
                "cumulative_clip_accuracy": float(end_row["cumulative_clip_accuracy"]),
                "cumulative_tda_accuracy": float(end_row["cumulative_tda_accuracy"]),
                "cumulative_tda_minus_clip_pp": float(
                    end_row["cumulative_tda_minus_clip_pp"]
                ),
                "cumulative_clip_correct_tda_wrong": int(
                    end_row["cumulative_clip_correct_tda_wrong"]
                ),
                "mean_positive_cache_purity": float(
                    interval_df["positive_cache_purity"].mean()
                ),
                "end_positive_cache_purity": float(
                    end_row["positive_cache_purity"]
                ),
                "mean_wrong_positive_cache_entries": float(
                    interval_df["wrong_positive_cache_entries"].mean()
                ),
                "end_wrong_positive_cache_entries": int(
                    end_row["wrong_positive_cache_entries"]
                ),
            }
        )

    return pd.DataFrame(rows)


def save_plots(
    interval_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    figure_dir: Path,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)

    for metric, title, ylabel, filename in [
        (
            "tda_interval_accuracy",
            "TDA interval accuracy under test ordering",
            "Accuracy (%)",
            "adverse_ordering_interval_accuracy.png",
        ),
        (
            "cumulative_tda_minus_clip_pp",
            "Cumulative TDA minus CLIP accuracy",
            "Accuracy difference (percentage points)",
            "adverse_ordering_cumulative_gap.png",
        ),
        (
            "end_positive_cache_purity",
            "Positive cache purity under test ordering",
            "Positive cache purity",
            "adverse_ordering_positive_cache_purity.png",
        ),
    ]:
        plt.figure(figsize=(10, 6))
        for run_name, run_df in interval_df.groupby("run_name", sort=False):
            plt.plot(
                run_df["end_step"],
                run_df[metric],
                marker="o",
                markersize=3,
                label=run_name,
            )
        plt.xlabel("Test stream step")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(figure_dir / filename, dpi=200)
        plt.close()

    ordered_summary = summary_df.sort_values(
        ["condition", "seed"],
        kind="stable",
    )
    plt.figure(figsize=(11, 6))
    plt.bar(ordered_summary["run_name"], ordered_summary["tda_minus_clip_pp"])
    plt.axhline(0.0, linewidth=1.0)
    plt.ylabel("TDA minus CLIP accuracy (percentage points)")
    plt.title("Final adaptation gain under test ordering")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(
        figure_dir / "adverse_ordering_final_accuracy_gap.png",
        dpi=200,
    )
    plt.close()


def log_run_to_wandb(
    run: ExperimentRun,
    config: dict[str, Any],
    interval_df: pd.DataFrame,
    summary: dict[str, Any],
    project: str,
) -> None:
    import wandb

    wandb_run = wandb.init(
        project=project,
        group="adverse-test-ordering",
        name=run.name,
        config={
            "condition": run.condition,
            "seed": run.seed,
            "stress_prefix_size": run.stress_prefix_size,
            "positive": config["positive"],
            "negative": config["negative"],
        },
        reinit=True,
    )

    for _, row in interval_df.iterrows():
        wandb_run.log(
            {
                "clip_interval_accuracy": row["clip_interval_accuracy"],
                "tda_interval_accuracy": row["tda_interval_accuracy"],
                "interval_tda_minus_clip_pp": row["interval_tda_minus_clip_pp"],
                "cumulative_clip_accuracy": row["cumulative_clip_accuracy"],
                "cumulative_tda_accuracy": row["cumulative_tda_accuracy"],
                "cumulative_tda_minus_clip_pp": row["cumulative_tda_minus_clip_pp"],
                "positive_cache_purity": row["end_positive_cache_purity"],
                "wrong_positive_cache_entries": row[
                    "end_wrong_positive_cache_entries"
                ],
            },
            step=int(row["end_step"]),
        )

    for key, value in summary.items():
        if isinstance(value, (int, float, str)):
            wandb_run.summary[key] = value

    wandb_run.finish()


def main() -> None:
    args = parse_arguments()

    feature_path = Path(args.features).resolve()
    config_path = Path(args.config).resolve()
    output_root = Path(args.output_root).resolve()

    if not feature_path.is_file():
        raise FileNotFoundError(f"Feature file was not found: {feature_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file was not found: {config_path}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use --device cpu or select a GPU runtime.")

    device = torch.device(args.device)
    artifact = load_torch_artifact(feature_path)
    config = load_config(config_path)

    raw_dir = output_root / "raw" / "adverse_ordering"
    summary_dir = output_root / "summaries"
    figure_dir = output_root / "figures"
    raw_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    runs = build_runs(
        artifact,
        args.conditions,
        args.random_seeds,
        args.deterministic_seed,
        args.stress_prefix_size,
    )

    if not runs:
        raise RuntimeError("No experiment runs were generated.")

    summary_rows: list[dict[str, Any]] = []
    interval_frames: list[pd.DataFrame] = []
    event_frames: list[pd.DataFrame] = []

    for run in runs:
        print(f"Running condition: {run.name}")
        sample_df, event_df, summary = simulate_run(
            artifact,
            config,
            run,
            device,
        )
        interval_df = build_interval_metrics(sample_df, args.interval_size)

        sample_path = raw_dir / f"{run.name}_samples.csv"
        event_path = raw_dir / f"{run.name}_positive_cache_events.csv"
        sample_df.to_csv(sample_path, index=False)
        event_df.to_csv(event_path, index=False)

        summary_rows.append(summary)
        interval_frames.append(interval_df)
        event_frames.append(event_df)

        print(f"  CLIP accuracy: {summary['clip_accuracy']:.2f}%")
        print(f"  TDA accuracy: {summary['tda_accuracy']:.2f}%")
        print(f"  TDA minus CLIP: {summary['tda_minus_clip_pp']:+.2f} points")
        print(
            "  CLIP-correct to TDA-wrong count:",
            summary["clip_correct_tda_wrong_count"],
        )
        print(
            "  Final positive cache purity:",
            f"{summary['final_positive_cache_purity']:.4f}",
        )

        if args.wandb_log:
            log_run_to_wandb(
                run,
                config,
                interval_df,
                summary,
                args.wandb_project,
            )

    summary_df = pd.DataFrame(summary_rows)
    all_intervals_df = pd.concat(interval_frames, ignore_index=True)
    all_events_df = pd.concat(event_frames, ignore_index=True)

    summary_path = summary_dir / "adverse_ordering_summary.csv"
    intervals_path = summary_dir / "adverse_ordering_intervals.csv"
    events_path = raw_dir / "adverse_ordering_all_positive_cache_events.csv"
    metadata_path = summary_dir / "adverse_ordering_metadata.json"

    summary_df.to_csv(summary_path, index=False)
    all_intervals_df.to_csv(intervals_path, index=False)
    all_events_df.to_csv(events_path, index=False)

    metadata = {
        "feature_file": str(feature_path),
        "config_file": str(config_path),
        "interval_size": args.interval_size,
        "random_seeds": args.random_seeds,
        "deterministic_seed": args.deterministic_seed,
        "requested_stress_prefix_size": args.stress_prefix_size,
        "conditions": args.conditions,
        "dataset": artifact["dataset"],
        "backbone": artifact["backbone"],
        "num_samples": len(artifact["labels"]),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    save_plots(all_intervals_df, summary_df, figure_dir)

    print("\nExperiment completed.")
    print("Summary CSV:", summary_path)
    print("Interval CSV:", intervals_path)
    print("Cache event CSV:", events_path)
    print("Figure directory:", figure_dir)


if __name__ == "__main__":
    main()
