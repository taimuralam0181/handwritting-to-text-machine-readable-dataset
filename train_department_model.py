from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "medicine_department_seed.csv"
MODEL_PATH = BASE_DIR / "models" / "medicine_department_classifier.pkl"


def train_department_model(dataset_path=DATASET_PATH, model_path=MODEL_PATH):
    dataset_path = Path(dataset_path)
    model_path = Path(model_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)
    df = df.dropna(subset=["medicine_name", "department"]).copy()
    df["medicine_name"] = df["medicine_name"].astype(str).str.strip()
    df["department"] = df["department"].astype(str).str.strip()
    df = df[df["medicine_name"] != ""]

    if df["department"].nunique() < 2:
        raise ValueError("Need at least two department labels to train the classifier.")

    X = df["medicine_name"]
    y = df["department"]

    model = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5))),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])

    report_text = ""
    if len(df) >= 10 and y.value_counts().min() >= 2:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        report_text = classification_report(y_test, predictions)
    else:
        model.fit(X, y)
        report_text = "Dataset too small for a stable split; trained on full dataset."

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    return {
        "model_path": str(model_path),
        "training_rows": int(len(df)),
        "labels": sorted(df["department"].unique()),
        "report_text": report_text,
    }


def main():
    result = train_department_model()
    print("Classification report:")
    print(result["report_text"])
    print(f"Saved model to: {result['model_path']}")
    print(f"Training rows: {result['training_rows']}")
    print(f"Labels: {result['labels']}")


if __name__ == "__main__":
    main()
