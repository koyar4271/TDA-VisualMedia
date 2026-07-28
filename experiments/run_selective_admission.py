from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import torch


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

import run_unknown_contamination as base


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare original TDA with cache-specific selective admission "
            "under clean and unknown-contaminated test streams."
        )
    )
    parser.add_argument("--known-features", required=True)
    parser.add_argument("--unknown-features", required=True)
    parser.add_argument("--config", default="./configs/caltech101.yaml")
    parser.add_argument("--output-root", default="./results")
    parser.add_argument(
        "--clean-known-order-seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
    )
    parser.add_argument(
        "--contamination-known-order-seed",
        type=int,
        default=0,
    )
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
        default=[25, 50, 100, 200],
    )
    parser.add_argument(
        "--unknown-seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
    )
    parser.add_argument("--calibration-size", type=int, default=100)
    parser.add_argument(
        "--pos-entropy-quantile",
        type=float,
        default=0.90,
        help=(
            "Quantile of normalized entropy in the calibration prefix used "
            "as tau_pos when --tau-pos is not specified."
        ),
    )
    parser.add_argument(
        "--similarity-quantile",
        type=float,
        default=0.05,
        help=(
            "Quantile of maximum cosine similarity in the calibration prefix "
            "used as tau_s when --tau-s is not specified."
        ),
    )
    parser.add_argument(
        "--tau-pos",
        type=float,
        default=None,
        help="Optional fixed positive-cache entropy threshold.",
    )
    parser.add_argument(
        "--tau-s",
        type=float,
        default=None,
        help="Optional fixed maximum-similarity threshold.",
    )
    parser.add_argument("--recovery-bin-size", type=int, default=50)
    parser.add_argument("--main-prefix-size", type=int, default=100)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    return parser.parse_args()


def validate_quantile(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], but received {value}.")


def calibrate_thresholds(
    known_artifact: dict[str, Any],
    known_order: list[int],
    calibration_size: int,
    pos_entropy_quantile: float,
    similarity_quantile: float,
    fixed_tau_pos: float | None,
    fixed_tau_s: float | None,
) -> tuple[float, float]:
    if calibration_size <= 0:
        raise ValueError("calibration_size must be positive.")
    if calibration_size > len(known_order):
        raise ValueError(
            "calibration_size exceeds the number of known samples."
        )

    calibration_indices = torch.tensor(
        known_order[:calibration_size],
        dtype=torch.long,
    )
    calibration_entropy = known_artifact["normalized_entropy"][
        calibration_indices
    ].float()
    calibration_similarity = known_artifact["max_similarity"][
        calibration_indices
    ].float()

    tau_pos = (
        float(fixed_tau_pos)
        if fixed_tau_pos is not None
        else float(
            torch.quantile(
                calibration_entropy,
                pos_entropy_quantile,
            ).item()
        )
    )
    tau_s = (
        float(fixed_tau_s)
        if fixed_tau_s is not None
        else float(
            torch.quantile(
                calibration_similarity,
                similarity_quantile,
            ).item()
        )
    )

    return tau_pos, tau_s


def simulate_selective_stream(
    known_artifact: dict[str, Any],
    unknown_artifact: dict[str, Any],
    config: dict[str, Any],
    stream: list[base.StreamItem],
    run_name: str,
    prefix_size: int,
    unknown_count: int,
    unknown_seed: int,
    device: torch.device,
    tau_pos: float,
    tau_s: float,
    admission_start_step: int,
) -> base.SimulationResult:
    num_classes = int(known_artifact["num_classes"])
    classnames = list(known_artifact["classnames"])
    labels = known_artifact["labels"]

    pos_cfg = config["positive"]
    neg_cfg = config["negative"]

    positive_cache: dict[int, list[base.TrackedCacheEntry]] = {}
    negative_cache: dict[int, list[base.TrackedCacheEntry]] = {}
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

    lower_entropy = float(
        neg_cfg["entropy_threshold"]["lower"]
    )
    upper_entropy = float(
        neg_cfg["entropy_threshold"]["upper"]
    )

    for stream_step, item in enumerate(stream):
        artifact = (
            known_artifact
            if item.origin == "known"
            else unknown_artifact
        )
        sample = base.get_sample_tensors(
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

        admission_active = stream_step >= admission_start_step
        similarity_passed = sample["max_similarity"] > tau_s
        positive_entropy_passed = (
            sample["normalized_entropy"] < tau_pos
        )
        negative_entropy_candidate = (
            lower_entropy
            < sample["normalized_entropy"]
            < upper_entropy
        )

        positive_gate_passed = (
            True
            if not admission_active
            else positive_entropy_passed and similarity_passed
        )
        negative_gate_passed = (
            negative_entropy_candidate
            and (
                True
                if not admission_active
                else similarity_passed
            )
        )

        positive_admitted = False
        negative_admitted = False
        positive_replaced = False
        negative_replaced = False

        if bool(pos_cfg["enabled"]) and positive_gate_passed:
            positive_entry = base.TrackedCacheEntry(
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
            positive_admitted, removed_entry = base.update_cache(
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
                base.finalize_unknown_event(
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

        if bool(neg_cfg["enabled"]) and negative_gate_passed:
            negative_entry = base.TrackedCacheEntry(
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
            negative_admitted, removed_entry = base.update_cache(
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
                base.finalize_unknown_event(
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

        cache_logits = base.compute_origin_split_logits(
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
                "admission_active": int(admission_active),
                "similarity_passed": int(similarity_passed),
                "positive_entropy_passed": int(
                    positive_entropy_passed
                ),
                "negative_entropy_candidate": int(
                    negative_entropy_candidate
                ),
                "positive_gate_passed": int(positive_gate_passed),
                "negative_gate_passed": int(negative_gate_passed),
                "positive_admitted": int(positive_admitted),
                "negative_admitted": int(negative_admitted),
                "positive_replaced": int(positive_replaced),
                "negative_replaced": int(negative_replaced),
                "positive_cache_size": len(
                    base.flatten_cache(positive_cache)
                ),
                "negative_cache_size": len(
                    base.flatten_cache(negative_cache)
                ),
                "positive_unknown_cache_entries": (
                    base.cache_unknown_count(positive_cache)
                ),
                "negative_unknown_cache_entries": (
                    base.cache_unknown_count(negative_cache)
                ),
                "positive_contamination_ratio": (
                    base.cache_contamination_ratio(positive_cache)
                ),
                "negative_contamination_ratio": (
                    base.cache_contamination_ratio(negative_cache)
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
        for entry in base.flatten_cache(cache):
            base.finalize_unknown_event(
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

    return base.SimulationResult(
        sample_df=pd.DataFrame(sample_rows),
        unknown_event_df=pd.DataFrame(event_rows),
        counters=counters,
    )


def post_calibration_rates(
    result: base.SimulationResult,
    admission_start_step: int,
    method: str,
    config: dict[str, Any],
) -> dict[str, float]:
    post = result.sample_df[
        result.sample_df["stream_step"] >= admission_start_step
    ].copy()

    lower_entropy = float(
        config["negative"]["entropy_threshold"]["lower"]
    )
    upper_entropy = float(
        config["negative"]["entropy_threshold"]["upper"]
    )

    output: dict[str, float] = {}

    for origin in ["known", "unknown"]:
        subset = post[post["origin"] == origin]

        if subset.empty:
            for key in [
                "positive_admission_rate",
                "negative_admission_rate",
                "positive_gate_acceptance_rate",
                "negative_gate_acceptance_rate",
                "positive_gate_rejection_rate",
                "negative_gate_rejection_rate",
            ]:
                output[f"{origin}_{key}"] = float("nan")
            continue

        if method == "selective":
            positive_gate = subset["positive_gate_passed"]
            negative_gate = subset["negative_gate_passed"]
        else:
            positive_gate = pd.Series(
                1,
                index=subset.index,
                dtype=float,
            )
            negative_gate = (
                (subset["normalized_entropy"] > lower_entropy)
                & (subset["normalized_entropy"] < upper_entropy)
            ).astype(float)

        output[f"{origin}_positive_admission_rate"] = float(
            subset["positive_admitted"].mean()
        )
        output[f"{origin}_negative_admission_rate"] = float(
            subset["negative_admitted"].mean()
        )
        output[f"{origin}_positive_gate_acceptance_rate"] = float(
            positive_gate.mean()
        )
        output[f"{origin}_negative_gate_acceptance_rate"] = float(
            negative_gate.mean()
        )
        output[f"{origin}_positive_gate_rejection_rate"] = float(
            1.0 - positive_gate.mean()
        )
        output[f"{origin}_negative_gate_rejection_rate"] = float(
            1.0 - negative_gate.mean()
        )

    return output


def clean_accuracy(
    result: base.SimulationResult,
    calibration_size: int,
) -> tuple[float, float]:
    known = result.sample_df[result.sample_df["origin"] == "known"]
    suffix = known[known["known_position"] >= calibration_size]
    return (
        100.0 * known["tda_correct"].mean(),
        100.0 * suffix["tda_correct"].mean(),
    )


def save_plots(
    comparison_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    figure_dir: Path,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)

    aggregated = (
        comparison_df.groupby(
            ["prefix_size", "unknown_count"],
            as_index=False,
        )
        .agg(
            original_effect_mean=(
                "original_contamination_effect_pp",
                "mean",
            ),
            original_effect_std=(
                "original_contamination_effect_pp",
                "std",
            ),
            selective_effect_mean=(
                "selective_contamination_effect_pp",
                "mean",
            ),
            selective_effect_std=(
                "selective_contamination_effect_pp",
                "std",
            ),
            original_positive_contamination=(
                "original_peak_positive_contamination_ratio",
                "mean",
            ),
            selective_positive_contamination=(
                "selective_peak_positive_contamination_ratio",
                "mean",
            ),
            selective_unknown_positive_rejection=(
                "selective_unknown_positive_gate_rejection_rate",
                "mean",
            ),
            selective_unknown_negative_rejection=(
                "selective_unknown_negative_gate_rejection_rate",
                "mean",
            ),
        )
        .fillna(0.0)
    )

    for prefix_size, prefix_df in aggregated.groupby(
        "prefix_size",
        sort=True,
    ):
        plt.figure(figsize=(9, 6))
        plt.errorbar(
            prefix_df["unknown_count"],
            prefix_df["original_effect_mean"],
            yerr=prefix_df["original_effect_std"],
            marker="o",
            capsize=4,
            label="Original TDA",
        )
        plt.errorbar(
            prefix_df["unknown_count"],
            prefix_df["selective_effect_mean"],
            yerr=prefix_df["selective_effect_std"],
            marker="s",
            capsize=4,
            label="Selective admission",
        )
        plt.axhline(0.0, linewidth=1.0)
        plt.xlabel("Number of inserted unknown samples")
        plt.ylabel("Paired suffix accuracy difference (percentage points)")
        plt.title(
            f"Contamination effect with known prefix {prefix_size}"
        )
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            figure_dir
            / f"selective_admission_accuracy_prefix_{prefix_size}.png",
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
            prefix_df["original_positive_contamination"],
            marker="o",
            label=f"Original, prefix = {prefix_size}",
        )
        plt.plot(
            prefix_df["unknown_count"],
            prefix_df["selective_positive_contamination"],
            marker="s",
            linestyle="--",
            label=f"Selective, prefix = {prefix_size}",
        )
    plt.xlabel("Number of inserted unknown samples")
    plt.ylabel("Peak positive-cache contamination ratio")
    plt.title("Positive-cache contamination before and after improvement")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(
        figure_dir / "selective_admission_positive_contamination.png",
        dpi=200,
    )
    plt.close()

    main_prefix = int(aggregated["prefix_size"].min())
    rejection_df = aggregated[
        aggregated["prefix_size"] == main_prefix
    ]
    plt.figure(figsize=(9, 6))
    plt.plot(
        rejection_df["unknown_count"],
        rejection_df["selective_unknown_positive_rejection"],
        marker="o",
        label="Positive-cache gate",
    )
    plt.plot(
        rejection_df["unknown_count"],
        rejection_df["selective_unknown_negative_rejection"],
        marker="s",
        label="Negative-cache gate",
    )
    plt.xlabel("Number of inserted unknown samples")
    plt.ylabel("Unknown gate rejection rate")
    plt.title(
        f"Unknown rejection by selective admission (prefix = {main_prefix})"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        figure_dir / "selective_admission_unknown_rejection.png",
        dpi=200,
    )
    plt.close()

    clean_aggregated = (
        clean_df.groupby("method", as_index=False)
        .agg(
            mean_accuracy=("suffix_accuracy", "mean"),
            std_accuracy=("suffix_accuracy", "std"),
        )
        .fillna(0.0)
    )
    plt.figure(figsize=(7, 5))
    plt.errorbar(
        clean_aggregated["method"],
        clean_aggregated["mean_accuracy"],
        yerr=clean_aggregated["std_accuracy"],
        marker="o",
        capsize=4,
        linestyle="none",
    )
    plt.ylabel("Clean random-stream suffix accuracy (%)")
    plt.title("Clean-condition performance check")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        figure_dir / "selective_admission_clean_random.png",
        dpi=200,
    )
    plt.close()


def main() -> None:
    args = parse_arguments()

    validate_quantile(
        "pos_entropy_quantile",
        args.pos_entropy_quantile,
    )
    validate_quantile(
        "similarity_quantile",
        args.similarity_quantile,
    )

    known_feature_path = Path(args.known_features).expanduser().resolve()
    unknown_feature_path = Path(
        args.unknown_features
    ).expanduser().resolve()
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
    known_artifact = base.load_torch_artifact(known_feature_path)
    unknown_artifact = base.load_torch_artifact(unknown_feature_path)
    base.validate_artifacts(known_artifact, unknown_artifact)
    config = base.load_config(config_path)

    known_sample_count = len(known_artifact["labels"])
    unknown_sample_count = len(unknown_artifact["features"])

    if args.calibration_size >= known_sample_count:
        raise ValueError(
            "calibration_size must be smaller than the known sample count."
        )

    prefix_sizes = sorted(set(args.prefix_sizes))
    unknown_counts = sorted(set(args.unknown_counts))
    unknown_seeds = sorted(set(args.unknown_seeds))
    clean_known_order_seeds = sorted(
        set(args.clean_known_order_seeds)
    )

    for prefix_size in prefix_sizes:
        if prefix_size < args.calibration_size:
            raise ValueError(
                "Each prefix size must be at least calibration_size."
            )
        if prefix_size >= known_sample_count:
            raise ValueError(
                f"Invalid prefix size: {prefix_size}"
            )

    for unknown_count in unknown_counts:
        if unknown_count <= 0:
            raise ValueError(
                "unknown_counts must contain positive values only."
            )
        if unknown_count > unknown_sample_count:
            raise ValueError(
                f"Unknown count {unknown_count} exceeds "
                f"the available count {unknown_sample_count}."
            )

    summary_dir = output_root / "summaries"
    raw_dir = output_root / "raw" / "selective_admission"
    figure_dir = output_root / "figures"
    summary_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    print("Device:", device)
    print("Known samples:", known_sample_count)
    print("Unknown samples:", unknown_sample_count)
    print("Calibration size:", args.calibration_size)

    clean_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []

    for known_order_seed in clean_known_order_seeds:
        print("Running clean seed:", known_order_seed)
        known_order = base.build_known_order(
            known_sample_count,
            known_order_seed,
        )
        tau_pos, tau_s = calibrate_thresholds(
            known_artifact,
            known_order,
            args.calibration_size,
            args.pos_entropy_quantile,
            args.similarity_quantile,
            args.tau_pos,
            args.tau_s,
        )
        threshold_rows.append(
            {
                "context": "clean",
                "known_order_seed": known_order_seed,
                "prefix_size": args.calibration_size,
                "tau_pos": tau_pos,
                "tau_s": tau_s,
            }
        )

        stream = base.build_control_stream(known_order)

        original_result = base.simulate_stream(
            known_artifact,
            unknown_artifact,
            config,
            stream,
            run_name=f"clean_original_seed_{known_order_seed}",
            prefix_size=args.calibration_size,
            unknown_count=0,
            unknown_seed=-1,
            device=device,
        )
        selective_result = simulate_selective_stream(
            known_artifact,
            unknown_artifact,
            config,
            stream,
            run_name=f"clean_selective_seed_{known_order_seed}",
            prefix_size=args.calibration_size,
            unknown_count=0,
            unknown_seed=-1,
            device=device,
            tau_pos=tau_pos,
            tau_s=tau_s,
            admission_start_step=args.calibration_size,
        )

        for method, result in [
            ("original", original_result),
            ("selective", selective_result),
        ]:
            full_accuracy, suffix_accuracy = clean_accuracy(
                result,
                args.calibration_size,
            )
            rates = post_calibration_rates(
                result,
                args.calibration_size,
                method,
                config,
            )
            clean_rows.append(
                {
                    "method": method,
                    "known_order_seed": known_order_seed,
                    "calibration_size": args.calibration_size,
                    "tau_pos": tau_pos,
                    "tau_s": tau_s,
                    "full_accuracy": full_accuracy,
                    "suffix_accuracy": suffix_accuracy,
                    **rates,
                }
            )

    contamination_known_order = base.build_known_order(
        known_sample_count,
        args.contamination_known_order_seed,
    )
    contamination_tau_pos, contamination_tau_s = calibrate_thresholds(
        known_artifact,
        contamination_known_order,
        args.calibration_size,
        args.pos_entropy_quantile,
        args.similarity_quantile,
        args.tau_pos,
        args.tau_s,
    )
    threshold_rows.append(
        {
            "context": "contamination",
            "known_order_seed": args.contamination_known_order_seed,
            "prefix_size": args.calibration_size,
            "tau_pos": contamination_tau_pos,
            "tau_s": contamination_tau_s,
        }
    )

    control_stream = base.build_control_stream(
        contamination_known_order
    )
    original_control = base.simulate_stream(
        known_artifact,
        unknown_artifact,
        config,
        control_stream,
        run_name="improvement_original_control",
        prefix_size=0,
        unknown_count=0,
        unknown_seed=-1,
        device=device,
    )
    selective_control = simulate_selective_stream(
        known_artifact,
        unknown_artifact,
        config,
        control_stream,
        run_name="improvement_selective_control",
        prefix_size=0,
        unknown_count=0,
        unknown_seed=-1,
        device=device,
        tau_pos=contamination_tau_pos,
        tau_s=contamination_tau_s,
        admission_start_step=args.calibration_size,
    )

    run_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    recovery_frames: list[pd.DataFrame] = []
    class_frames: list[pd.DataFrame] = []

    for prefix_size in prefix_sizes:
        for unknown_count in unknown_counts:
            for unknown_seed in unknown_seeds:
                print(
                    "Running contamination:",
                    prefix_size,
                    unknown_count,
                    unknown_seed,
                )

                unknown_indices = base.select_unknown_indices(
                    unknown_sample_count,
                    unknown_count,
                    unknown_seed,
                )
                stream = base.build_contaminated_stream(
                    contamination_known_order,
                    prefix_size,
                    unknown_indices,
                )

                original_result = base.simulate_stream(
                    known_artifact,
                    unknown_artifact,
                    config,
                    stream,
                    run_name=(
                        f"original_prefix_{prefix_size}_"
                        f"unknown_{unknown_count}_seed_{unknown_seed}"
                    ),
                    prefix_size=prefix_size,
                    unknown_count=unknown_count,
                    unknown_seed=unknown_seed,
                    device=device,
                )
                selective_result = simulate_selective_stream(
                    known_artifact,
                    unknown_artifact,
                    config,
                    stream,
                    run_name=(
                        f"selective_prefix_{prefix_size}_"
                        f"unknown_{unknown_count}_seed_{unknown_seed}"
                    ),
                    prefix_size=prefix_size,
                    unknown_count=unknown_count,
                    unknown_seed=unknown_seed,
                    device=device,
                    tau_pos=contamination_tau_pos,
                    tau_s=contamination_tau_s,
                    admission_start_step=args.calibration_size,
                )

                method_summaries: dict[str, dict[str, Any]] = {}

                for method, control, result in [
                    (
                        "original",
                        original_control,
                        original_result,
                    ),
                    (
                        "selective",
                        selective_control,
                        selective_result,
                    ),
                ]:
                    paired_df = base.build_paired_suffix(
                        control.sample_df,
                        result.sample_df,
                        prefix_size,
                    )
                    paired_df.insert(0, "method", method)
                    paired_df.insert(1, "prefix_size", prefix_size)
                    paired_df.insert(2, "unknown_count", unknown_count)
                    paired_df.insert(3, "unknown_seed", unknown_seed)

                    summary = base.summarize_run(
                        paired_df,
                        result,
                        result.unknown_event_df,
                        run_name=(
                            f"{method}_prefix_{prefix_size}_"
                            f"unknown_{unknown_count}_seed_{unknown_seed}"
                        ),
                        known_order_seed=(
                            args.contamination_known_order_seed
                        ),
                        prefix_size=prefix_size,
                        unknown_count=unknown_count,
                        unknown_seed=unknown_seed,
                    )
                    summary["method"] = method
                    summary["tau_pos"] = (
                        contamination_tau_pos
                        if method == "selective"
                        else float("nan")
                    )
                    summary["tau_s"] = (
                        contamination_tau_s
                        if method == "selective"
                        else float("nan")
                    )
                    summary.update(
                        post_calibration_rates(
                            result,
                            args.calibration_size,
                            method,
                            config,
                        )
                    )
                    run_rows.append(summary)
                    method_summaries[method] = summary

                    recovery = pd.DataFrame(
                        base.build_recovery_rows(
                            paired_df,
                            run_name=summary["run_name"],
                            prefix_size=prefix_size,
                            unknown_count=unknown_count,
                            unknown_seed=unknown_seed,
                            bin_size=args.recovery_bin_size,
                        )
                    )
                    if not recovery.empty:
                        recovery.insert(0, "method", method)
                        recovery_frames.append(recovery)

                    class_effects = pd.DataFrame(
                        base.build_class_rows(
                            paired_df,
                            run_name=summary["run_name"],
                            prefix_size=prefix_size,
                            unknown_count=unknown_count,
                            unknown_seed=unknown_seed,
                        )
                    )
                    if not class_effects.empty:
                        class_effects.insert(0, "method", method)
                        class_frames.append(class_effects)

                original_summary = method_summaries["original"]
                selective_summary = method_summaries["selective"]

                comparison_rows.append(
                    {
                        "prefix_size": prefix_size,
                        "unknown_count": unknown_count,
                        "unknown_seed": unknown_seed,
                        "known_order_seed": (
                            args.contamination_known_order_seed
                        ),
                        "tau_pos": contamination_tau_pos,
                        "tau_s": contamination_tau_s,
                        "original_control_suffix_accuracy": (
                            original_summary[
                                "control_suffix_accuracy"
                            ]
                        ),
                        "selective_control_suffix_accuracy": (
                            selective_summary[
                                "control_suffix_accuracy"
                            ]
                        ),
                        "clean_accuracy_difference_pp": (
                            selective_summary[
                                "control_suffix_accuracy"
                            ]
                            - original_summary[
                                "control_suffix_accuracy"
                            ]
                        ),
                        "original_contaminated_suffix_accuracy": (
                            original_summary[
                                "contaminated_suffix_accuracy"
                            ]
                        ),
                        "selective_contaminated_suffix_accuracy": (
                            selective_summary[
                                "contaminated_suffix_accuracy"
                            ]
                        ),
                        "contaminated_accuracy_gain_pp": (
                            selective_summary[
                                "contaminated_suffix_accuracy"
                            ]
                            - original_summary[
                                "contaminated_suffix_accuracy"
                            ]
                        ),
                        "original_contamination_effect_pp": (
                            original_summary[
                                "paired_accuracy_difference_pp"
                            ]
                        ),
                        "selective_contamination_effect_pp": (
                            selective_summary[
                                "paired_accuracy_difference_pp"
                            ]
                        ),
                        "mitigation_pp": (
                            selective_summary[
                                "paired_accuracy_difference_pp"
                            ]
                            - original_summary[
                                "paired_accuracy_difference_pp"
                            ]
                        ),
                        "original_peak_positive_contamination_ratio": (
                            original_summary[
                                "peak_positive_contamination_ratio"
                            ]
                        ),
                        "selective_peak_positive_contamination_ratio": (
                            selective_summary[
                                "peak_positive_contamination_ratio"
                            ]
                        ),
                        "original_peak_negative_contamination_ratio": (
                            original_summary[
                                "peak_negative_contamination_ratio"
                            ]
                        ),
                        "selective_peak_negative_contamination_ratio": (
                            selective_summary[
                                "peak_negative_contamination_ratio"
                            ]
                        ),
                        "original_unknown_positive_admission_rate": (
                            original_summary[
                                "unknown_positive_admission_rate"
                            ]
                        ),
                        "selective_unknown_positive_admission_rate": (
                            selective_summary[
                                "unknown_positive_admission_rate"
                            ]
                        ),
                        "original_unknown_negative_admission_rate": (
                            original_summary[
                                "unknown_negative_admission_rate"
                            ]
                        ),
                        "selective_unknown_negative_admission_rate": (
                            selective_summary[
                                "unknown_negative_admission_rate"
                            ]
                        ),
                        "selective_unknown_positive_gate_rejection_rate": (
                            selective_summary[
                                "unknown_positive_gate_rejection_rate"
                            ]
                        ),
                        "selective_unknown_negative_gate_rejection_rate": (
                            selective_summary[
                                "unknown_negative_gate_rejection_rate"
                            ]
                        ),
                        "original_known_positive_admission_rate": (
                            original_summary[
                                "known_positive_admission_rate"
                            ]
                        ),
                        "selective_known_positive_admission_rate": (
                            selective_summary[
                                "known_positive_admission_rate"
                            ]
                        ),
                        "original_known_negative_admission_rate": (
                            original_summary[
                                "known_negative_admission_rate"
                            ]
                        ),
                        "selective_known_negative_admission_rate": (
                            selective_summary[
                                "known_negative_admission_rate"
                            ]
                        ),
                        "original_harmful_flip_count": (
                            original_summary[
                                "unknown_harmful_flip_count"
                            ]
                        ),
                        "selective_harmful_flip_count": (
                            selective_summary[
                                "unknown_harmful_flip_count"
                            ]
                        ),
                    }
                )

    clean_df = pd.DataFrame(clean_rows)
    thresholds_df = pd.DataFrame(threshold_rows)
    runs_df = pd.DataFrame(run_rows)
    comparison_df = pd.DataFrame(comparison_rows)
    recovery_df = (
        pd.concat(recovery_frames, ignore_index=True)
        if recovery_frames
        else pd.DataFrame()
    )
    class_df = (
        pd.concat(class_frames, ignore_index=True)
        if class_frames
        else pd.DataFrame()
    )

    clean_path = summary_dir / "selective_admission_clean_random.csv"
    thresholds_path = (
        summary_dir / "selective_admission_thresholds.csv"
    )
    runs_path = summary_dir / "selective_admission_runs.csv"
    comparison_path = (
        summary_dir / "selective_admission_comparison.csv"
    )
    recovery_path = (
        summary_dir / "selective_admission_recovery.csv"
    )
    class_path = (
        summary_dir / "selective_admission_class_effects.csv"
    )

    clean_df.to_csv(clean_path, index=False)
    thresholds_df.to_csv(thresholds_path, index=False)
    runs_df.to_csv(runs_path, index=False)
    comparison_df.to_csv(comparison_path, index=False)
    recovery_df.to_csv(recovery_path, index=False)
    class_df.to_csv(class_path, index=False)

    metadata = {
        "known_feature_path": str(known_feature_path),
        "unknown_feature_path": str(unknown_feature_path),
        "config_path": str(config_path),
        "clean_known_order_seeds": clean_known_order_seeds,
        "contamination_known_order_seed": (
            args.contamination_known_order_seed
        ),
        "prefix_sizes": prefix_sizes,
        "unknown_counts": unknown_counts,
        "unknown_seeds": unknown_seeds,
        "calibration_size": args.calibration_size,
        "admission_start_step": args.calibration_size,
        "pos_entropy_quantile": args.pos_entropy_quantile,
        "similarity_quantile": args.similarity_quantile,
        "fixed_tau_pos": args.tau_pos,
        "fixed_tau_s": args.tau_s,
        "contamination_tau_pos": contamination_tau_pos,
        "contamination_tau_s": contamination_tau_s,
        "positive_rule": (
            "Before calibration: original TDA. After calibration: "
            "normalized_entropy < tau_pos and max_similarity > tau_s."
        ),
        "negative_rule": (
            "Before calibration: original TDA. After calibration: "
            "tau_l < normalized_entropy < tau_h and "
            "max_similarity > tau_s."
        ),
        "threshold_calibration": (
            "Thresholds are calibrated without labels from the first "
            "calibration_size known samples of each known stream."
        ),
    }
    metadata_path = (
        summary_dir / "selective_admission_metadata.json"
    )
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    save_plots(
        comparison_df,
        clean_df,
        figure_dir,
    )

    print("Saved:", clean_path)
    print("Saved:", thresholds_path)
    print("Saved:", runs_path)
    print("Saved:", comparison_path)
    print("Saved:", recovery_path)
    print("Saved:", class_path)
    print("Saved:", metadata_path)
    print("Figures:", figure_dir)


if __name__ == "__main__":
    main()
