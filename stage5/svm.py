import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import PowerTransformer
import joblib

# 1. Load and Feature Engineering
df = pd.read_csv(r'stage5\input\stage5_input.csv')

# Engineering extra non-linear features
df['tilt_velocity_impact'] = df['max_tilt'] * abs(df['weighted_velocity'])
df['low_hwr_near_ground'] = (1 / (df['min_h_w_ratio'] + 0.1)) * (1 - df['weighted_ground_proximity'])
df['depth_velocity_sync'] = df['depth_drop'] * df['weighted_velocity']
df['tilt_hwr_ratio'] = df['max_tilt'] / (df['min_h_w_ratio'] + 0.1)

features = [
    'max_tilt', 'delta_tilt', 'min_h_w_ratio', 'delta_h_w_ratio',
    'weighted_tilt', 'weighted_velocity', 'weighted_ground_proximity',
    'depth_drop', 'depth_variance', 'depth_range',
    'tilt_velocity_impact', 'low_hwr_near_ground', 'depth_velocity_sync', 'tilt_hwr_ratio'
]

X = df[features]
y = df['LABEL']

# 2. Advanced Scaling: PowerTransformer
# Unlike StandardScaler, this maps data to a normal distribution, 
# which helps SVM find a cleaner "hyperplane".
scaler = PowerTransformer(method='yeo-johnson')
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# 3. Hyperparameter Tuning with GridSearchCV
# We test different C and Gamma values to find the "Sweet Spot"
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001, 'scale'],
    'kernel': ['rbf']
}

print("Searching for optimal SVM parameters (this may take a minute)...")
grid = GridSearchCV(SVC(probability=True, random_state=42), param_grid, refit=True, verbose=1, cv=5)
grid.fit(X_train, y_train)

# 4. Results
best_svm = grid.best_estimator_
y_pred = best_svm.predict(X_test)

print(f"\nBEST PARAMS: {grid.best_params_}")
print(f"IMPROVED SVM ACCURACY: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

# 5. Save the tuned model
joblib.dump(best_svm, 'stage5_svm_model.pkl')
joblib.dump(scaler, 'stage5_svm_scaler.pkl')
print("Optimized SVM Assets saved.")