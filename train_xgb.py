#!/usr/bin/env python3
import argparse, json, os, inspect
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score, roc_auc_score
from joblib import dump
from xgboost import XGBClassifier

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="creditcard.csv")
    p.add_argument("--outdir", default="models")
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.data)
    if "Class" not in df.columns:
        raise SystemExit("could not find 'Class' column in the csv")

    y = df["Class"].astype(int)
    X = df.drop(columns=["Class"])
    cont_feats = [c for c in ["Amount", "Time"] if c in X.columns]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pre = ColumnTransformer(
        [("scale", StandardScaler(), cont_feats)],
        remainder="passthrough"
    )
    Xtr = pre.fit_transform(X_train)
    Xva = pre.transform(X_val)

    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    spw = (neg / pos) if pos > 0 else 1.0

    clf = XGBClassifier(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        min_child_weight=1,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
        scale_pos_weight=spw
    )

    # --- early stopping that works across xgboost versions ---
    sig = inspect.signature(clf.fit).parameters
    fit_kwargs = {"eval_set": [(Xva, y_val)], "verbose": False}
    if "callbacks" in sig:
        from xgboost import callback as xgb_callback
        fit_kwargs["callbacks"] = [xgb_callback.EarlyStopping(rounds=50, save_best=True)]
    elif "early_stopping_rounds" in sig:
        fit_kwargs["early_stopping_rounds"] = 50
    # else: train full n_estimators with no early stopping

    clf.fit(Xtr, y_train, **fit_kwargs)

    # evaluate
    p_val = clf.predict_proba(Xva)[:, 1]
    ap = average_precision_score(y_val, p_val)
    roc = roc_auc_score(y_val, p_val)
    print(f"validation AUPRC: {ap:.4f} | ROC-AUC: {roc:.4f}")

    # best iteration if available
    best_iter = None
    try:
        best_iter = getattr(clf.get_booster(), "best_iteration", None)
    except Exception:
        pass
    print(f"best_iteration: {best_iter if best_iter is not None else 'n/a'}")

    # save artifacts
    model_path = os.path.join(args.outdir, "xgb_model.json")
    pre_path = os.path.join(args.outdir, "preprocessor.joblib")
    meta_path = os.path.join(args.outdir, "metadata.json")
    clf.save_model(model_path)
    dump(pre, pre_path)
    with open(meta_path, "w") as f:
        json.dump({
            "original_columns": X.columns.tolist(),
            "scaled_columns": cont_feats,
            "best_iteration": int(best_iter) if isinstance(best_iter, int) else None,
            "scale_pos_weight": float(spw)
        }, f, indent=2)

    print(f"saved: {model_path}\n       {pre_path}\n       {meta_path}")

if __name__ == "__main__":
    main()
