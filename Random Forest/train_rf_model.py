import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib

def main():
    print("Loading data...")
    df = pd.read_csv('urban_flood_risk_dataset.csv')
    
    # 2. Preprocess the Data identically so our Map Generators don't break
    X = df.drop(['flood_risk_score', 'flood_risk_category'], axis=1)
    y = df['flood_risk_category']
    X = pd.get_dummies(X, columns=['soil_type'], drop_first=False)
    
    # 3. Split the Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Hyperparameter Tuning using Grid Search Cross Validation
    print("Initiating Grid Search Optimization. This will test hundreds of forest architectures...")
    
    base_model = RandomForestClassifier(random_state=42, class_weight='balanced')
    
    # We feed it a massive grid of options to mathematically test against each other
    param_grid = {
        'n_estimators': [100, 200, 400],         # Number of trees
        'max_depth': [10, 15, 25, None],         # Max levels in each tree
        'min_samples_split': [2, 5],             # Min data points required to split a node
        'min_samples_leaf': [1, 2],              # Min data points allowed at a leaf node
        'max_features': ['sqrt', 'log2']         # Number of features to consider
    }
    
    # cv=3 means it splits the training data 3 times to cross-validate every parameter
    grid_search = GridSearchCV(estimator=base_model, param_grid=param_grid, 
                               cv=3, n_jobs=-1, verbose=1, scoring='accuracy')
    
    grid_search.fit(X_train, y_train)
    
    print(f"\nOptimization Complete! Best parameters found: {grid_search.best_params_}")
    
    # 5. Extract the absolute best model from the Grid Search
    best_rf_model = grid_search.best_estimator_
    
    print("Evaluating mathematically optimized model on unseen test data...")
    y_pred = best_rf_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print("\n--- UPGRADED MODEL RESULTS ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}\n")
    
    print("--- Detailed Classification Report ---")
    print(classification_report(y_test, y_pred))
    
    # 7. Overwrite the old file with the new elite model
    joblib.dump(best_rf_model, 'random_forest_flood_model.pkl')
    print("Elite Optimized Model saved, overriding the old 'random_forest_flood_model.pkl'!")

if __name__ == "__main__":
    main()
