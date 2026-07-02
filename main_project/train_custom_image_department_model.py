from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
TRAINING_DATASET_DIR = BASE_DIR / "training_dataset"
DEFAULT_MODEL_DIR = BASE_DIR / "models" / "custom_image_department_classifier"
IMAGE_SIZE = (160, 48)

CUSTOM_SOURCES = [
    TRAINING_DATASET_DIR / "cardiology_custom_20260625_200149" / "line_crops",
    TRAINING_DATASET_DIR / "ophthalmology_custom_20260625_202639" / "line_crops",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train image-based department classifier from custom line-crop images.")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--image-width", type=int, default=IMAGE_SIZE[0])
    parser.add_argument("--image-height", type=int, default=IMAGE_SIZE[1])
    parser.add_argument("--max-iter", type=int, default=300)
    return parser.parse_args()


def load_labeled_rows() -> pd.DataFrame:
    frames = []
    for source_dir in CUSTOM_SOURCES:
        labels_path = source_dir / "labels.csv"
        images_dir = source_dir / "images"
        if not labels_path.exists():
            raise FileNotFoundError(f"Missing labels file: {labels_path}")
        if not images_dir.exists():
            raise FileNotFoundError(f"Missing images folder: {images_dir}")

        df = pd.read_csv(labels_path).fillna("")
        required = {"crop_image_name", "medicine_name", "department"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{labels_path} missing columns: {sorted(missing)}")

        df = df[["crop_image_name", "medicine_name", "department"]].copy()
        df["image_path"] = df["crop_image_name"].apply(lambda name: images_dir / str(name).strip())
        df["medicine_name"] = df["medicine_name"].astype(str).str.strip()
        df["department"] = df["department"].astype(str).str.strip()
        df = df[(df["medicine_name"] != "") & (df["department"] != "")]
        df = df[df["image_path"].apply(lambda path: Path(path).exists())]
        frames.append(df)

    if not frames:
        raise ValueError("No labeled image rows found.")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["image_path"], keep="first")
    if combined["department"].nunique() < 2:
        raise ValueError("Need at least two departments for training.")
    return combined


def preprocess_image(image_path: Path, image_size: tuple[int, int]) -> np.ndarray:
    image = Image.open(image_path).convert("L")
    image = ImageOps.autocontrast(image)
    image = image.resize(image_size, Image.Resampling.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = 1.0 - arr
    return arr.flatten()


def build_features(df: pd.DataFrame, image_size: tuple[int, int]) -> np.ndarray:
    features = [preprocess_image(Path(path), image_size) for path in df["image_path"]]
    return np.asarray(features, dtype=np.float32)


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir).resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    image_size = (args.image_width, args.image_height)

    df = load_labeled_rows()
    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=20260702,
        stratify=df["department"],
    )
    validation_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=20260702,
        stratify=temp_df["department"],
    )

    x_train = build_features(train_df, image_size)
    x_validation = build_features(validation_df, image_size)
    x_test = build_features(test_df, image_size)

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train_df["department"])
    y_validation = encoder.transform(validation_df["department"])
    y_test = encoder.transform(test_df["department"])

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=False)),
            (
                "classifier",
                SGDClassifier(
                    loss="log_loss",
                    penalty="l2",
                    alpha=1e-5,
                    max_iter=args.max_iter,
                    tol=1e-3,
                    random_state=20260702,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)

    validation_pred = pipeline.predict(x_validation)
    test_pred = pipeline.predict(x_test)

    labels = list(encoder.classes_)
    metrics = {
        "total_labeled_line_crop_images": int(len(df)),
        "train_examples": int(len(train_df)),
        "validation_examples": int(len(validation_df)),
        "test_examples": int(len(test_df)),
        "labels": labels,
        "image_size": list(image_size),
        "validation_accuracy": float(accuracy_score(y_validation, validation_pred)),
        "test_accuracy": float(accuracy_score(y_test, test_pred)),
        "validation_confusion_matrix": confusion_matrix(y_validation, validation_pred).tolist(),
        "test_confusion_matrix": confusion_matrix(y_test, test_pred).tolist(),
        "source_folders": [str(path.relative_to(BASE_DIR)) for path in CUSTOM_SOURCES],
    }

    joblib.dump(
        {
            "pipeline": pipeline,
            "label_encoder": encoder,
            "image_size": image_size,
            "labels": labels,
        },
        model_dir / "custom_image_department_classifier.joblib",
    )
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (model_dir / "validation_report.txt").write_text(
        classification_report(y_validation, validation_pred, target_names=labels, zero_division=0),
        encoding="utf-8",
    )
    (model_dir / "test_report.txt").write_text(
        classification_report(y_test, test_pred, target_names=labels, zero_division=0),
        encoding="utf-8",
    )

    preview = test_df[["image_path", "medicine_name", "department"]].copy().reset_index(drop=True)
    preview["predicted_department"] = encoder.inverse_transform(test_pred)
    preview["match"] = preview["department"] == preview["predicted_department"]
    preview["image_path"] = preview["image_path"].apply(lambda path: str(Path(path).relative_to(BASE_DIR)))
    preview.head(50).to_csv(model_dir / "prediction_preview.csv", index=False)

    print(json.dumps(metrics, indent=2))
    print(f"Saved model: {model_dir / 'custom_image_department_classifier.joblib'}")


if __name__ == "__main__":
    main()
