from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocess_functions.trial_classification import (
    TrialFeaturizer,
    load_labels,
    load_trial_catalog,
    make_aligned_loader,
    predict_confidence,
)


def main() -> None:
    args = parse_args()

    try:
        import joblib
    except ImportError as exc:
        raise ImportError(
            "joblib is required to load the saved sklearn model. "
            "Install joblib in this environment."
        ) from exc

    output_root = Path(args.output_root)
    classification_dir = output_root / "trial_classification"
    model_path = Path(args.model)
    predictions_out = (
        Path(args.predictions_out)
        if args.predictions_out
        else classification_dir / "saved_model_trial_predictions.csv"
    )
    labels_csv = Path(args.labels_csv) if args.labels_csv else classification_dir / "trial_labels.csv"

    bundle = joblib.load(model_path)
    model = bundle["model"]
    model_name = bundle.get("model_name", model.__class__.__name__)

    catalog, _, _ = load_trial_catalog(output_root)
    catalog = filter_catalog(catalog, args.only)
    if catalog.empty:
        raise ValueError(f"No trials matched --only {args.only!r}")

    load_aligned_for_row = make_aligned_loader()
    featurizer = TrialFeaturizer(
        fixed_t=int(bundle.get("fixed_t", args.fixed_t)),
        fps=float(bundle.get("fps", args.fps)),
    )
    featurizer.apply_bundle_schema(bundle)
    features = featurizer.build_matrix(
        catalog,
        load_aligned_for_row=load_aligned_for_row,
        skipped_path=classification_dir / "saved_model_feature_skipped_trials.csv",
    )

    pred_label = model.predict(features.X)
    pred_conf = predict_confidence(model, features.X)

    predictions = features.meta.copy()
    predictions["pred_label"] = pred_label
    predictions["pred_conf"] = pred_conf
    predictions["model_name"] = model_name
    predictions["model_path"] = str(model_path)

    labels = load_labels(labels_csv)
    if not labels.empty:
        predictions = predictions.merge(
            labels[["recording_id", "trial_idx", "label"]].rename(
                columns={"label": "manual_label"}
            ),
            on=["recording_id", "trial_idx"],
            how="left",
        )
    else:
        predictions["manual_label"] = pd.NA

    predictions_out.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(predictions_out, index=False)

    index = pd.DataFrame(
        [
            {
                "model_path": str(model_path),
                "model_name": model_name,
                "predictions_csv": str(predictions_out),
                "n_trials": len(predictions),
                "n_recordings": predictions["recording_id"].nunique(),
                "labels_csv": str(labels_csv),
                "status": "predicted",
            }
        ]
    )
    index_path = classification_dir / "saved_model_prediction_index.csv"
    index.to_csv(index_path, index=False)

    print(f"Loaded model: {model_path}")
    print(f"Model name: {model_name}")
    print(f"Predicted {len(predictions)} trials from {predictions['recording_id'].nunique()} recordings")
    print(f"Wrote {predictions_out}")
    print(f"Wrote {index_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict segmented trial labels with a saved sklearn model bundle."
    )
    parser.add_argument("--output-root", default="preprocess_out")
    parser.add_argument(
        "--model",
        default="ML_model/trial_type_classifier.joblib",
        help="Saved model bundle from 20260923_trial_classification.ipynb",
    )
    parser.add_argument("--predictions-out")
    parser.add_argument("--labels-csv")
    parser.add_argument(
        "--only",
        help="Recording ID, session ID, mouse ID, trial type, or comma-separated list",
    )
    parser.add_argument("--fixed-t", type=int, default=200)
    parser.add_argument("--fps", type=float, default=30.0)
    return parser.parse_args()


def filter_catalog(catalog: pd.DataFrame, only: str | None) -> pd.DataFrame:
    if only is None:
        return catalog
    requested = {item.strip() for item in str(only).split(",") if item.strip()}
    cols = [col for col in ["recording_id", "session_id", "mouse_id", "trial_type"] if col in catalog.columns]
    mask = pd.Series(False, index=catalog.index)
    for col in cols:
        mask |= catalog[col].astype(str).isin(requested)
    return catalog[mask].reset_index(drop=True)


if __name__ == "__main__":
    main()
