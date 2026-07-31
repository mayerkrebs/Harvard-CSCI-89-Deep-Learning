"""Create a deterministic representative sample of the processed XDD_015 data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import shutil
from collections import defaultdict
from pathlib import Path, PurePosixPath


SEED = 42
GENUS_COUNT = 3
SPECIMENS_PER_GENUS = 2
IMAGES_PER_SPECIMEN = 2
EXPECTED_IMAGE_COUNT = GENUS_COUNT * SPECIMENS_PER_GENUS * IMAGES_PER_SPECIMEN
MAX_FOLDER_BYTES = 10 * 1024 * 1024
OFFICIAL_DATASET_URL = (
    "https://repository.kulib.kyoto-u.ac.jp/items/"
    "7f0f28bd-f9b0-4603-be8e-7c7d908a245f"
)


def find_default_source(script_dir: Path) -> Path:
    candidates = [script_dir / "wood_data", script_dir.parent / "wood_data"]
    for candidate in candidates:
        if (candidate / "manifest.csv").is_file():
            return candidate
    return candidates[0]


def read_manifest(manifest_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with manifest_path.open(newline="", encoding="utf-8-sig") as manifest_file:
        reader = csv.DictReader(manifest_file)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    required = {"filepath", "genus", "specimen_id", "split", "image_index", "h5_path"}
    missing = required.difference(fieldnames)
    if missing:
        raise ValueError(f"Manifest is missing required fields: {sorted(missing)}")
    return fieldnames, rows


def select_rows(rows: list[dict[str, str]]) -> tuple[list[str], list[dict[str, str]]]:
    rng = random.Random(SEED)
    specimen_rows: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    specimen_splits: dict[tuple[str, str], set[str]] = defaultdict(set)

    for row in rows:
        key = (row["genus"], row["specimen_id"])
        specimen_rows[key].append(row)
        specimen_splits[key].add(row["split"])

    eligible_by_genus: dict[str, list[str]] = defaultdict(list)
    for (genus, specimen_id), available_rows in specimen_rows.items():
        if len(available_rows) >= IMAGES_PER_SPECIMEN and len(specimen_splits[(genus, specimen_id)]) == 1:
            eligible_by_genus[genus].append(specimen_id)

    eligible_genera = sorted(
        genus
        for genus, specimen_ids in eligible_by_genus.items()
        if len(specimen_ids) >= SPECIMENS_PER_GENUS
    )
    if len(eligible_genera) < GENUS_COUNT:
        raise ValueError(f"Need at least {GENUS_COUNT} eligible genera")

    selected_genera = sorted(rng.sample(eligible_genera, GENUS_COUNT))
    selected_specimens: dict[str, list[str]] | None = None

    for _ in range(10_000):
        candidate = {
            genus: rng.sample(sorted(eligible_by_genus[genus]), SPECIMENS_PER_GENUS)
            for genus in selected_genera
        }
        selected_splits = {
            next(iter(specimen_splits[(genus, specimen_id)]))
            for genus, specimen_ids in candidate.items()
            for specimen_id in specimen_ids
        }
        if selected_splits == {"train", "validation", "test"}:
            selected_specimens = candidate
            break

    if selected_specimens is None:
        raise ValueError("Could not select six specimens spanning train, validation, and test")

    selected_rows: list[dict[str, str]] = []
    for genus in selected_genera:
        for specimen_id in sorted(selected_specimens[genus]):
            candidates = sorted(
                specimen_rows[(genus, specimen_id)],
                key=lambda row: (int(row["image_index"]), row["filepath"]),
            )
            selected_rows.extend(rng.sample(candidates, IMAGES_PER_SPECIMEN))

    selected_rows.sort(
        key=lambda row: (
            row["genus"],
            row["specimen_id"],
            int(row["image_index"]),
        )
    )
    return selected_genera, selected_rows


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_readme(output_dir: Path, selected_rows: list[dict[str, str]]) -> None:
    summary = sorted(
        {
            (row["genus"], row["specimen_id"], row["split"])
            for row in selected_rows
        }
    )
    summary_rows = "\n".join(
        f"| {genus} | {specimen_id} | {split} | {IMAGES_PER_SPECIMEN} |"
        for genus, specimen_id, split in summary
    )
    readme = f"""# Representative Processed Dataset Sample

This is a representative sample of the processed XDD_015 image dataset. It is intended only to demonstrate the image format, folder organization, manifest structure, and loading workflow.

This sample is not sufficient to train or reproduce the reported 17-genus model results. Reproduction of the complete analysis requires downloading XDD_015 from the [official Kyoto University repository]({OFFICIAL_DATASET_URL}) and running `1. data_pipeline.ipynb`.

The sample was generated from `wood_data/manifest.csv` with `create_sample_dataset.py` using random seed {SEED}. It contains three genera, two specimens per genus, and two unmodified PNG images per specimen. Original specimen identities, split assignments, image indices, and HDF5 provenance are retained in `sample_manifest.csv`.

## Included Specimens

| Genus | Specimen ID | Original split | Images |
|---|---|---|---:|
{summary_rows}

## Loading Example

```python
from pathlib import Path
import pandas as pd
from PIL import Image

sample_dir = Path("sample_data")
manifest = pd.read_csv(sample_dir / "sample_manifest.csv")
image = Image.open(sample_dir / manifest.loc[0, "filepath"])
print(manifest.head())
print(image.mode, image.size)
```
"""
    with (output_dir / "README.md").open("w", encoding="utf-8", newline="\n") as readme_file:
        readme_file.write(readme)


def publish_sample(temporary_dir: Path, output_dir: Path) -> None:
    if not output_dir.exists():
        temporary_dir.replace(output_dir)
        return

    expected_files = {
        path.relative_to(temporary_dir)
        for path in temporary_dir.rglob("*")
        if path.is_file()
    }
    for existing_path in output_dir.rglob("*"):
        if existing_path.is_file() and existing_path.relative_to(output_dir) not in expected_files:
            existing_path.unlink()

    for source_path in temporary_dir.rglob("*"):
        if source_path.is_file():
            relative_path = source_path.relative_to(temporary_dir)
            destination_path = output_dir / relative_path
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination_path)

    for directory in sorted(
        (path for path in output_dir.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        try:
            directory.rmdir()
        except OSError:
            pass
    shutil.rmtree(temporary_dir)


def build_sample(source_dir: Path, output_dir: Path) -> tuple[list[str], list[dict[str, str]], int]:
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    manifest_path = source_dir / "manifest.csv"
    fieldnames, rows = read_manifest(manifest_path)
    selected_genera, selected_rows = select_rows(rows)

    temporary_dir = output_dir.with_name(f".{output_dir.name}_tmp")
    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)
    temporary_dir.mkdir(parents=True)

    output_rows: list[dict[str, str]] = []
    source_by_output_path: dict[str, Path] = {}
    try:
        for row in selected_rows:
            source_path = source_dir / Path(row["filepath"])
            if not source_path.is_file():
                raise FileNotFoundError(f"Source image does not exist: {source_path}")

            relative_path = PurePosixPath(
                "images", row["split"], row["genus"], source_path.name
            ).as_posix()
            destination_path = temporary_dir / Path(relative_path)
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination_path)

            output_row = dict(row)
            output_row["filepath"] = relative_path
            output_rows.append(output_row)
            source_by_output_path[relative_path] = source_path

        with (temporary_dir / "sample_manifest.csv").open(
            "w", newline="", encoding="utf-8"
        ) as manifest_file:
            writer = csv.DictWriter(manifest_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(output_rows)

        write_readme(temporary_dir, output_rows)

        copied_images = sorted((temporary_dir / "images").rglob("*.png"))
        genera = {row["genus"] for row in output_rows}
        specimens = {(row["genus"], row["specimen_id"]) for row in output_rows}
        splits = {row["split"] for row in output_rows}
        specimen_split_counts: dict[tuple[str, str], set[str]] = defaultdict(set)
        for row in output_rows:
            specimen_split_counts[(row["genus"], row["specimen_id"])].add(row["split"])
            copied_path = temporary_dir / Path(row["filepath"])
            if not copied_path.is_file():
                raise FileNotFoundError(f"Manifest path does not exist: {copied_path}")
            if file_digest(copied_path) != file_digest(source_by_output_path[row["filepath"]]):
                raise ValueError(f"Copied image differs from source: {row['filepath']}")

        total_bytes = sum(path.stat().st_size for path in temporary_dir.rglob("*") if path.is_file())
        if len(output_rows) != EXPECTED_IMAGE_COUNT or len(copied_images) != EXPECTED_IMAGE_COUNT:
            raise ValueError(f"Expected exactly {EXPECTED_IMAGE_COUNT} copied images")
        if len(genera) != GENUS_COUNT or len(specimens) != GENUS_COUNT * SPECIMENS_PER_GENUS:
            raise ValueError("Sample does not contain exactly three genera and six specimens")
        if splits != {"train", "validation", "test"}:
            raise ValueError(f"Sample does not span all required splits: {sorted(splits)}")
        if any(len(specimen_splits) != 1 for specimen_splits in specimen_split_counts.values()):
            raise ValueError("A selected specimen appears under more than one split")
        if total_bytes >= MAX_FOLDER_BYTES:
            raise ValueError(f"Sample folder is not below 10 MiB: {total_bytes:,} bytes")

        publish_sample(temporary_dir, output_dir)
    except Exception:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
        raise

    return selected_genera, output_rows, total_bytes


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=find_default_source(script_dir),
        help="Processed wood_data directory containing manifest.csv (default: auto-detect)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "sample_data",
        help="Sample output directory (default: sample_data beside this script)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_genera, selected_rows, total_bytes = build_sample(args.source_dir, args.output_dir)
    specimens = sorted(
        {
            (row["genus"], row["specimen_id"], row["split"])
            for row in selected_rows
        }
    )

    print("Representative sample created and validated")
    print(f"Selected genera ({len(selected_genera)}): {', '.join(selected_genera)}")
    print("Selected specimens and original splits:")
    for genus, specimen_id, split in specimens:
        print(f"  {genus}: {specimen_id} ({split})")
    print(f"Splits represented: {', '.join(sorted({row['split'] for row in selected_rows}))}")
    print(f"Image count: {len(selected_rows)}")
    print(f"Total folder size: {total_bytes / (1024 * 1024):.2f} MiB ({total_bytes:,} bytes)")


if __name__ == "__main__":
    main()