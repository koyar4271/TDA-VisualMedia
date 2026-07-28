from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import clip
from datasets import build_dataset
from datasets.utils import read_image
from utils import clip_classifier


class IndexedImageDataset(Dataset):
    """Return a transformed image together with its label and metadata."""

    def __init__(self, data_source: list[Any], transform: Any) -> None:
        self.data_source = data_source
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data_source)

    def __getitem__(self, index: int):
        item = self.data_source[index]
        image = read_image(item.impath)
        image = self.transform(image)
        return image, item.label, index, item.impath


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute CLIP features for TDA ordering experiments."
    )
    parser.add_argument("--dataset", default="caltech101")
    parser.add_argument("--data-root", default="./dataset")
    parser.add_argument("--backbone", default="RN50", choices=["RN50", "ViT-B/16"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not args.overwrite:
        print(f"Feature file already exists: {output_path}")
        print("Use --overwrite to recompute it.")
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Select a GPU runtime in Google Colab.")

    device = torch.device("cuda")
    print("Device:", torch.cuda.get_device_name(0))
    print("Dataset:", args.dataset)
    print("Backbone:", args.backbone)

    clip_model, preprocess = clip.load(args.backbone, device=device)
    clip_model.eval()

    dataset = build_dataset(args.dataset, args.data_root)
    classnames = dataset.classnames
    template = dataset.template
    clip_weights = clip_classifier(classnames, template, clip_model)

    indexed_dataset = IndexedImageDataset(dataset.test, preprocess)
    loader = DataLoader(
        indexed_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    feature_batches: list[torch.Tensor] = []
    logit_batches: list[torch.Tensor] = []
    probability_batches: list[torch.Tensor] = []
    raw_entropy_batches: list[torch.Tensor] = []
    normalized_entropy_batches: list[torch.Tensor] = []
    prediction_batches: list[torch.Tensor] = []
    max_similarity_batches: list[torch.Tensor] = []
    probability_margin_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []
    index_batches: list[torch.Tensor] = []
    image_paths: list[str] = []

    max_entropy_denominator = math.log2(len(classnames))

    with torch.no_grad():
        for images, labels, indices, paths in tqdm(loader, desc="Precomputing CLIP features"):
            images = images.to(device, non_blocking=True)

            image_features = clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            cosine_similarities = image_features @ clip_weights
            clip_logits = 100.0 * cosine_similarities
            probabilities = clip_logits.softmax(dim=1)
            raw_entropy = -(probabilities * clip_logits.log_softmax(dim=1)).sum(dim=1)
            normalized_entropy = raw_entropy / max_entropy_denominator
            predictions = clip_logits.argmax(dim=1)
            max_similarity = cosine_similarities.max(dim=1).values

            top_values = probabilities.topk(k=min(2, probabilities.shape[1]), dim=1).values
            if top_values.shape[1] == 2:
                probability_margin = top_values[:, 0] - top_values[:, 1]
            else:
                probability_margin = torch.full_like(top_values[:, 0], float("nan"))

            feature_batches.append(image_features.detach().cpu().half())
            logit_batches.append(clip_logits.detach().cpu().half())
            probability_batches.append(probabilities.detach().cpu().half())
            raw_entropy_batches.append(raw_entropy.detach().cpu().float())
            normalized_entropy_batches.append(normalized_entropy.detach().cpu().float())
            prediction_batches.append(predictions.detach().cpu().long())
            max_similarity_batches.append(max_similarity.detach().cpu().float())
            probability_margin_batches.append(probability_margin.detach().cpu().float())
            label_batches.append(labels.detach().cpu().long())
            index_batches.append(indices.detach().cpu().long())
            image_paths.extend(str(path) for path in paths)

    features = torch.cat(feature_batches, dim=0)
    clip_logits = torch.cat(logit_batches, dim=0)
    probabilities = torch.cat(probability_batches, dim=0)
    raw_entropy = torch.cat(raw_entropy_batches, dim=0)
    normalized_entropy = torch.cat(normalized_entropy_batches, dim=0)
    predictions = torch.cat(prediction_batches, dim=0)
    max_similarity = torch.cat(max_similarity_batches, dim=0)
    probability_margin = torch.cat(probability_margin_batches, dim=0)
    labels = torch.cat(label_batches, dim=0)
    original_indices = torch.cat(index_batches, dim=0)

    clip_accuracy = 100.0 * (predictions == labels).float().mean().item()

    artifact = {
        "dataset": args.dataset,
        "backbone": args.backbone,
        "features": features,
        "clip_logits": clip_logits,
        "probabilities": probabilities,
        "raw_entropy": raw_entropy,
        "normalized_entropy": normalized_entropy,
        "clip_predictions": predictions,
        "max_similarity": max_similarity,
        "probability_margin": probability_margin,
        "labels": labels,
        "original_indices": original_indices,
        "image_paths": image_paths,
        "classnames": list(classnames),
        "template": list(template),
        "num_classes": len(classnames),
        "clip_accuracy": clip_accuracy,
        "official_entropy_normalization": "raw_softmax_entropy_divided_by_log2_num_classes",
    }

    torch.save(artifact, output_path)

    print("Number of samples:", len(labels))
    print(f"CLIP accuracy: {clip_accuracy:.2f}%")
    print("Feature shape:", tuple(features.shape))
    print("Saved feature file:", output_path)


if __name__ == "__main__":
    main()
