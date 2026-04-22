import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

MODEL_PATH = 'model.pkl'

def get_dataset_path():
    """Tries to find the Kaggle dataset, downloading it if needed."""
    # Try kagglehub auto-download
    try:
        import kagglehub
        path = kagglehub.dataset_download("harshitpathak18/hand-sign-dataset")
        csv_path = os.path.join(path, "sign_data.csv")
        if os.path.exists(csv_path):
            return csv_path
    except Exception as e:
        print(f"kagglehub download failed: {e}")

    # Fallback: search common cache locations
    possible_roots = [
        os.path.expanduser("~/.cache/kagglehub/datasets/harshitpathak18/hand-sign-dataset"),
        os.path.expanduser("~/Downloads"),
        ".",
    ]
    for root in possible_roots:
        for dirpath, _, filenames in os.walk(root):
            if "sign_data.csv" in filenames:
                return os.path.join(dirpath, "sign_data.csv")

    return None

def train_model():
    dataset_path = get_dataset_path()

    if not dataset_path:
        print("Could not find sign_data.csv. Please download the Kaggle dataset manually.")
        print("Dataset: https://www.kaggle.com/datasets/harshitpathak18/hand-sign-dataset")
        return

    print(f"Loading dataset from: {dataset_path}")
    try:
        df = pd.read_csv(dataset_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    if len(df) == 0:
        print("Dataset is empty.")
        return

    df.dropna(inplace=True)

    # Kaggle CSV has 210 pairwise distance features + 'Sign' label column
    X = df.drop('Sign', axis=1)
    y = df['Sign']

    # Normalize each sample by its max value (scale-invariant)
    X = X.div(X.max(axis=1), axis=0)

    if len(np.unique(y)) < 2:
        print(f"Only {len(np.unique(y))} class(es) found. Need at least 2 to train.")
        return

    print(f"Classes found: {sorted(np.unique(y))}")
    print(f"Total samples: {len(df)}")

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    print(f"Training Random Forest on {len(X_train)} samples...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)

    print(f"\nModel saved to '{MODEL_PATH}'")

if __name__ == '__main__':
    train_model()