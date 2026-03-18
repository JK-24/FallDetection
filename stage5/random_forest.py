import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import RobustScaler # Switched to RobustScaler to handle outliers
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load Data
df = pd.read_csv(r'stage5\input\stage5_input.csv')

# 2. Features (Same as before + 1 extra stability feature)
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

# 3. Scaling with RobustScaler (better for physics data with spikes)
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# 4. The "Final Push" Ensemble
# Using more estimators and a slightly slower learning rate for better generalization
rf = RandomForestClassifier(n_estimators=500, max_depth=16, max_features='sqrt', random_state=42)
hgb = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.02, max_depth=12, l2_regularization=1.5, random_state=42)

ensemble_model = VotingClassifier(
    estimators=[('rf', rf), ('hgb', hgb)],
    voting='soft',
    weights=[1, 2] # Giving more weight to HGB as it handles non-linear physics better
)

print("Training Final Optimized Ensemble...")
ensemble_model.fit(X_train, y_train)

# 5. Evaluation
y_pred = ensemble_model.predict(X_test)
print(f"\nNEW ACCURACY: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))


# Run this immediately after Section 5 in your existing code
results_df = pd.DataFrame(scaler.inverse_transform(X_test), columns=features)
results_df['Actual'] = y_test.values
results_df['Predicted'] = y_pred

# Filter for misclassifications
misclassified = results_df[results_df['Actual'] != results_df['Predicted']]

# Save for manual review
misclassified.to_csv('failed_cases_audit.csv', index=False)
print(f"Found {len(misclassified)} misclassified samples. Exported to failed_cases_audit.csv")

# Find the most confusing features
sns.boxplot(x='Actual', y='max_tilt', data=results_df)
plt.title('Max Tilt Distribution: Fall vs No Fall')
plt.show()


# 6. Save
joblib.dump(ensemble_model, 'stage5_physics_model.pkl')
joblib.dump(scaler, 'stage5_scaler.pkl')