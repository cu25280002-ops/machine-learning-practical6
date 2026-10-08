import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
np.random.seed(42)
n_samples = 300
attendance = np.random.uniform(50, 100, n_samples)
internal_marks = np.random.uniform(10, 50, n_samples)
assignments = np.random.uniform(5, 20, n_samples)
study_hours = np.random.uniform(1, 10, n_samples)
prev_sem_marks = np.random.uniform(40, 100, n_samples)
final_score = (
    0.2 * attendance
    + 0.3 * internal_marks
    + 0.1 * assignments
    + 1.5 * study_hours
    + 0.35 * prev_sem_marks
    + np.random.normal(0, 3, n_samples)
)
final_score = np.clip(final_score, 0, 100)
data = pd.DataFrame(
    {
        'Attendance': attendance,
        'Internal_Marks': internal_marks,
        'Assignments': assignments,
        'Study_Hours': study_hours,
        'Prev_Sem_Marks': prev_sem_marks,
        'Final_Score': final_score,
    }
)
X = data.drop('Final_Score', axis=1)
y = data['Final_Score']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest Regressor': RandomForestRegressor(
        n_estimators=100, random_state=42
    ),
}
results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    results[name] = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2 Score': r2}
results_df = pd.DataFrame(results).T
print("=== Project 1: Model Evaluation Metrics ===")
print(results_df)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.heatmap(data.corr(), annot=True, cmap='Blues', fmt='.2f')
plt.title('Feature Correlation Matrix')
best_model = models['Random Forest Regressor']
y_pred_rf = best_model.predict(X_test_scaled)
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_rf, alpha=0.7, color='teal')
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--',
    lw=2,
)
plt.xlabel('Actual Final Score')
plt.ylabel('Predicted Final Score')
plt.title('Actual vs Predicted (Random Forest)')
plt.tight_layout()
plt.show()
joblib.dump(best_model, 'student_performance_model.pkl')
joblib.dump(scaler, 'student_scaler.pkl')
print("\nModel saved as 'student_performance_model.pkl'")