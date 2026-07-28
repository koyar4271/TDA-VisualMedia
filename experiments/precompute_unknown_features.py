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
from datasets.utils import read_image
from utils import clip_classifier


class ImagePathDataset(Dataset):
    """Load images from a list of paths and apply the CLIP transform."""

    def __init__(self, image_paths: list[Path], transform: Any) -> None:
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        image = read_image(str(image_path))
        image = self.transform(image)
        return image, index, str(image_path)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute CLIP features for unknown Caltech101 background images."
        )
    )
    parser.add_argument("--known-features", required=True)
    parser.add_argument("--data-root", default="./dataset")
    parser.add_argument(
        "--unknown-directory",
        default=None,
        help=(
            "Directory containing unknown images. By default, the script uses "
            "Caltech101/101_ObjectCategories/BACKGROUND_Google."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_torch_artifact(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def find_unknown_directory(data_root: Path, requested: str | None) -> Path:
    if requested is not None:
        unknown_directory = Path(requested).expanduser().resolve()
    else:
        unknown_directory = (
            data_root
            / "caltech-101"
            / "101_ObjectCategories"
            / "BACKGROUND_Google"
        ).resolve()

    if not unknown_directory.is_dir():
        raise FileNotFoundError(
            f"The unknown image directory was not found: {unknown_directory}"
        )

    return unknown_directory


def collect_image_paths(directory: Path) -> list[Path]:
    supported_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in supported_extensions
    )

    if not image_paths:
        raise RuntimeError(
            f"No supported image files were found in: {directory}"
        )

    return image_paths


def main() -> None:
    args = parse_arguments()

    known_feature_path = Path(args.known_features).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not args.overwrite:
        print(f"Unknown feature file already exists: {output_path}")
        print("Use --overwrite to recompute it.")
        return

    if not known_feature_path.is_file():
        raise FileNotFoundError(
            f"The known feature file was not found: {known_feature_path}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Select a GPU runtime in Google Colab."
        )

    known_artifact = load_torch_artifact(known_feature_path)
    required_keys = {
        "backbone",
        "classnames",
        "template",
        "num_classes",
    }
    missing_keys = sorted(required_keys.difference(known_artifact))
    if missing_keys:
        raise KeyError(
            "The known feature artifact is missing required keys: "
            + ", ".join(missing_keys)
        )

    data_root = Path(args.data_root).expanduser().resolve()
    unknown_directory = find_unknown_directory(
        data_root,
        args.unknown_directory,
    )
    image_paths = collect_image_paths(unknown_directory)

    backbone = str(known_artifact["backbone"])
    classnames = list(known_artifact["classnames"])
    template = list(known_artifact["template"])
    num_classes = int(known_artifact["num_classes"])

    if len(classnames) != num_classes:
        raise ValueError(
            "The number of class names does not match num_classes."
        )

    device = torch.device("cuda")
    print("Device:", torch.cuda.get_device_name(0))
    print("Backbone:", backbone)
    print("Unknown directory:", unknown_directory)
    print("Unknown image count:", len(image_paths))

    clip_model, preprocess = clip.load(backbone, device=device)
    clip_model.eval()
    clip_weights = clip_classifier(
        classnames,
        template,
        clip_model,
    )

    dataset = ImagePathDataset(image_paths, preprocess)
    loader = DataLoader(
        dataset,
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
    original_index_batches: list[torch.Tensor] = []
    ordered_image_paths: list[str] = []

    max_entropy_denominator = math.log2(num_classes)

    with torch.no_grad():
        for images, indices, paths in tqdm(
            loader,
            desc="Precomputing unknown CLIP features",
        ):
            images = images.to(device, non_blocking=True)

            image_features = clip_model.encode_image(images)
            image_features = image_features / image_features.norm(
                dim=-1,
                keepdim=True,
            )

            cosine_similarities = image_features @ clip_weights
            clip_logits = 100.0 * cosine_similarities
            probabilities = clip_logits.softmax(dim=1)
            raw_entropy = -(
                probabilities * clip_logits.log_softmax(dim=1)
            ).sum(dim=1)
            normalized_entropy = raw_entropy / max_entropy_denominator
            predictions = clip_logits.argmax(dim=1)
            max_similarity = cosine_similarities.max(dim=1).values

            top_values = probabilities.topk(
                k=min(2, probabilities.shape[1]),
                dim=1,
            ).values
            if top_values.shape[1] == 2:
                probability_margin = top_values[:, 0] - top_values[:, 1]
            else:
                probability_margin = torch.full_like(
                    top_values[:, 0],
                    float("nan"),
                )

            feature_batches.append(image_features.detach().cpu().half())
            logit_batches.append(clip_logits.detach().cpu().half())
            probability_batches.append(probabilities.detach().cpu().half())
            raw_entropy_batches.append(raw_entropy.detach().cpu().float())
            normalized_entropy_batches.append(
                normalized_entropy.detach().cpu().float()
            )
            prediction_batches.append(predictions.detach().cpu().long())
            max_similarity_batches.append(
                max_similarity.detach().cpu().float()
            )
            probability_margin_batches.append(
                probability_margin.detach().cpu().float()
            )
            original_index_batches.append(indices.detach().cpu().long())
            ordered_image_paths.extend(str(path) for path in paths)

    artifact = {
        "source_name": "BACKGROUND_Google",
        "source_directory": str(unknown_directory),
        "backbone": backbone,
        "features": torch.cat(feature_batches, dim=0),
        "clip_logits": torch.cat(logit_batches, dim=0),
        "probabilities": torch.cat(probability_batches, dim=0),
        "raw_entropy": torch.cat(raw_entropy_batches, dim=0),
        "normalized_entropy": torch.cat(
            normalized_entropy_batches,
            dim=0,
        ),
        "clip_predictions": torch.cat(prediction_batches, dim=0),
        "max_similarity": torch.cat(max_similarity_batches, dim=0),
        "probability_margin": torch.cat(
            probability_margin_batches,
            dim=0,
        ),
        "original_indices": torch.cat(
            original_index_batches,
            dim=0,
        ),
        "image_paths": ordered_image_paths,
        "classnames": classnames,
        "template": template,
        "num_classes": num_classes,
        "official_entropy_normalization": (
            "raw_softmax_entropy_divided_by_log2_num_classes"
        ),
    }

    torch.save(artifact, output_path)

    print("Unknown sample count:", len(ordered_image_paths))
    print("Feature shape:", tuple(artifact["features"].shape))
    print("Saved unknown feature file:", output_path)


if __name__ == "__main__":
    main()
