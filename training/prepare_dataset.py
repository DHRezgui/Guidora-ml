"""CLI to prepare ML datasets from backend exports.

Examples:
    python ml/training/prepare_dataset.py --input ml/data/raw/dataset.json
    python ml/training/prepare_dataset.py --input ml/data/raw/dataset.csv --output ml/data/processed
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Allow direct execution: python ml/training/prepare_dataset.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.training.preprocess import load_exported_dataset, prepare_dataset, split_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare dataset and create engineered features")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to backend exported dataset (.json or .csv)",
    )
    parser.add_argument(
        "--output",
        default="ml/data/processed",
        help="Output directory for processed and split datasets",
    )
    return parser.parse_args()


def save_outputs(output_dir: Path, prepared, splits) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    prepared.frame.to_csv(output_dir / "dataset_prepared.csv", index=False)

    splits.x_train.to_csv(output_dir / "X_train.csv", index=False)
    splits.x_valid.to_csv(output_dir / "X_valid.csv", index=False)
    splits.x_test.to_csv(output_dir / "X_test.csv", index=False)

    splits.y_train.to_frame(name=prepared.target_column).to_csv(output_dir / "y_train.csv", index=False)
    splits.y_valid.to_frame(name=prepared.target_column).to_csv(output_dir / "y_valid.csv", index=False)
    splits.y_test.to_frame(name=prepared.target_column).to_csv(output_dir / "y_test.csv", index=False)

    metadata = {
        "target_column": prepared.target_column,
        "feature_columns": prepared.feature_columns,
        "rows": len(prepared.frame),
        "train_rows": len(splits.x_train),
        "valid_rows": len(splits.x_valid),
        "test_rows": len(splits.x_test),
    }

    with (output_dir / "dataset_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    df = load_exported_dataset(input_path)
    prepared = prepare_dataset(df)
    splits = split_dataset(prepared)
    save_outputs(output_dir, prepared, splits)

    print("Dataset preparation completed")
    print(f"Input rows: {len(df)}")
    print(f"Prepared rows: {len(prepared.frame)}")
    print(f"Output folder: {output_dir}")
    print(f"Features: {len(prepared.feature_columns)}")


if __name__ == "__main__":
    main()
