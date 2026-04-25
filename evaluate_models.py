import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

def load_data():
    # Load the upgraded dataset from the Random Forest folder
    df = pd.read_csv('Random Forest/urban_flood_risk_dataset.csv')
    X = df.drop(['flood_risk_score', 'flood_risk_category'], axis=1)
    y = df['flood_risk_category']
    # One-hot encode soil_type
    X = pd.get_dummies(X, columns=['soil_type'], drop_first=False)
    return X, y

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    # For ROC AUC we need probability estimates; handle if not available
    try:
        y_proba = model.predict_proba(X_test)
        roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
    except Exception:
        roc_auc = None
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    print(f"--- {model_name} ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    if roc_auc is not None:
        print(f"ROC AUC  : {roc_auc:.4f}")
    else:
        print("ROC AUC  : Not available for this model")
    print()

def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # Random Forest (use same hyperparameters as before for consistency)
    rf = RandomForestClassifier(n_estimators=400, max_depth=25, min_samples_leaf=2, min_samples_split=5, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)
    evaluate_model(rf, X_test, y_test, "Random Forest")
    # Ridge Classifier (with scaling)
    ridge = make_pipeline(StandardScaler(), RidgeClassifier(alpha=1.0))
    ridge.fit(X_train, y_train)
    evaluate_model(ridge, X_test, y_test, "Ridge Classifier")
    # MLP Classifier (simple architecture)
    mlp = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42))
    mlp.fit(X_train, y_train)
    evaluate_model(mlp, X_test, y_test, "MLP Classifier")

if __name__ == "__main__":
    main()
