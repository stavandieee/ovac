import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, classification_report

def main(data_path="data/feas_dataset.npz", out_path="coordination/models/feas_mlp.joblib"):
    d = np.load(data_path)
    X, y = d["X"], d["y"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

    clf = MLPClassifier(
        hidden_layer_sizes=(64, 64),
        activation="relu",
        alpha=1e-4,
        learning_rate_init=1e-3,
        batch_size=512,
        max_iter=30,
        random_state=0,
        verbose=True,
    )
    clf.fit(Xtr, ytr)

    p = clf.predict_proba(Xte)[:, 1]
    print("AUC:", roc_auc_score(yte, p))
    print(classification_report(yte, (p > 0.5).astype(int)))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(clf, out_path)
    print("Saved model:", out_path)

if __name__ == "__main__":
    main()
