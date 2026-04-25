import pandas as pd
import joblib
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings('ignore')

def main():
    print("1. Loading Pristine Dataset...")
    df = pd.read_csv('Random Forest/urban_flood_risk_dataset.csv')
    # DROP lat and lon so the AI learns universal physics rather than strictly geofencing to the specific training region coordinates.
    X = df.drop(['flood_risk_score', 'flood_risk_category', 'lat', 'lon'], axis=1, errors='ignore')
    y = df['flood_risk_category']
    X = pd.get_dummies(X, columns=['soil_type'], drop_first=False)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("2. Formulating Independent Base Models...")
    rf_model = RandomForestClassifier(n_estimators=400, max_depth=25, min_samples_leaf=2, min_samples_split=5, random_state=42, class_weight='balanced')
    # Use CalibratedClassifierCV so Ridge outputs proper math probabilities required for soft-voting!
    ridge_model = make_pipeline(StandardScaler(), CalibratedClassifierCV(RidgeClassifier(alpha=1.0)))
    mlp_model = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42))
    
    print("3. Assembling the Master Pipeline (Soft Voting Container)...")
    ensemble = VotingClassifier(
        estimators=[('rf', rf_model), ('ridge', ridge_model), ('mlp', mlp_model)],
        voting='soft'
    )
    
    print("4. Algorithmic GridSearch: Calculating Mathematically Optimal Fractional Weights...")
    # Generate weight variations (fractions jumping by 10% that add perfectly to 100%)
    weights_list = []
    for i in range(1, 10):
        for j in range(1, 10):
            for k in range(1, 10):
                if i + j + k == 10:
                    weights_list.append([round(i/10.0, 1), round(j/10.0, 1), round(k/10.0, 1)])
    
    param_grid = {'weights': weights_list}
    grid_search = GridSearchCV(estimator=ensemble, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    best_weights = grid_search.best_params_['weights']
    print(f"\n✅ Optimization Concluded! Exact fractional weights assigned:")
    print(f"  -> Random Forest Model : {round(best_weights[0] * 100, 1)}%")
    print(f"  -> Ridge Model         : {round(best_weights[1] * 100, 1)}%")
    print(f"  -> MLP Neural Network  : {round(best_weights[2] * 100, 1)}%")
    
    best_ensemble = grid_search.best_estimator_
    
    print("\n5. Testing Final Master Pipeline on Unseen Data...")
    y_pred = best_ensemble.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print("\n--- 🏆 UNIFIED SYSTEM METRICS 🏆 ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}\n")
    
    output_path = 'Random Forest/unified_weighted_ensemble.pkl'
    joblib.dump(best_ensemble, output_path)
    print(f"Unified Master AI Pipeline saved entirely within: '{output_path}'")

if __name__ == "__main__":
    main()
