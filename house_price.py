import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
np.random.seed(42)
n_samples = 400
area = np.random.uniform(500, 3500, n_samples)
bedrooms = np.random.randint(1, 6, n_samples)
bathrooms = np.random.randint(1, 4, n_samples)
parking = np.random.randint(0, 3, n_samples)
age = np.random.uniform(1, 30, n_samples)
location = np.random.choice(['Urban', 'Suburban', 'Rural'], n_samples)
location_encoded = pd.get_dummies(location, prefix='Loc', drop_first=True)
loc_multiplier = np.where(
    location == 'Urban', 25, np.where(location == 'Suburban', 15, 5)
)
price = (
    area * 0.05
    + bedrooms * 5
    + bathrooms * 4
    + parking * 3
    - age * 0.5
    + loc_multiplier
    + np.random.normal(0, 10, n_samples)
)
data = pd.DataFrame(
    {
        'Area': area,
        'Bedrooms': bedrooms,
        'Bathrooms': bathrooms,
        'Parking': parking,
        'Age': age,
        'Location': location,
        'Price': price,
    }
)
df_processed = pd.get_dummies(data, columns=['Location'], drop_first=True)
X = df_processed.drop('Price', axis=1)
y = df_processed['Price']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
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
print("=== Project 2: Model Evaluation Metrics ===")
print(results_df)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.barplot(x=results_df.index, y=results_df['R2 Score'], palette='viridis')
plt.title('Model R2 Score Comparison')
plt.ylim(0, 1)
rf_model = models['Random Forest']
importances = rf_model.feature_importances_
features = X.columns
plt.subplot(1, 2, 2)
sns.barplot(x=importances, y=features, palette='magma')
plt.title('Feature Importances (Random Forest)')
plt.tight_layout()
plt.show()
joblib.dump(rf_model, 'house_price_model.pkl')
joblib.dump(scaler, 'house_scaler.pkl')
print("\nModel saved as 'house_price_model.pkl'")