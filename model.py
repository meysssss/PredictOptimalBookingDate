import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import KFold, cross_validate, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
import optuna
from xgboost import XGBRegressor

# --------------------------------------------------
# 1. Load and Prepare Data
# --------------------------------------------------
df = pd.read_csv('cleaned_flight_booking_data_train.csv')

# Drop unused columns (dates & raw route)
df['Booking Date'] = pd.to_datetime(df['Booking Date'])
df['Departure Date'] = pd.to_datetime(df['Departure Date'])
df['Arrival Date'] = pd.to_datetime(df['Arrival Date'])

X = df.drop(columns=['Price', 'Booking Date', 'Departure Date', 'Arrival Date', 'Route'])
y = df['Price']

# --------------------------------------------------
# 2. Cross-validation setup
# --------------------------------------------------
cv = KFold(n_splits=5, shuffle=True, random_state=42)
scoring = {
    'MAE': 'neg_mean_absolute_error',
    'RMSE': 'neg_root_mean_squared_error',
    'R2': 'r2'
}

# --------------------------------------------------
# 3. Decision Tree
# --------------------------------------------------
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X, y)
joblib.dump(dt, 'decision_tree_model.pkl')

# --------------------------------------------------
# 4. Random Forest (GridSearchCV)
# --------------------------------------------------
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 4]
}
rf_grid = GridSearchCV(RandomForestRegressor(random_state=42), rf_params, cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
rf_grid.fit(X, y)
best_rf = rf_grid.best_estimator_
joblib.dump(best_rf, 'random_forest_model.pkl')

# --------------------------------------------------
# 5. XGBoost (Optuna)
# --------------------------------------------------
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0)
    }
    model = XGBRegressor(**params, objective='reg:squarederror', use_label_encoder=False, random_state=42)
    scores = cross_validate(model, X, y, cv=3, scoring={'RMSE': 'neg_root_mean_squared_error'})
    return -scores['test_RMSE'].mean()

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50, show_progress_bar=True)
xgb_best = XGBRegressor(**study.best_params, objective='reg:squarederror', use_label_encoder=False, random_state=42)
xgb_best.fit(X, y)
joblib.dump(xgb_best, 'xgboost_model.pkl')

# --------------------------------------------------
# 6. Stacking Regressor
# --------------------------------------------------
stack_model = StackingRegressor(
    estimators=[
        ('rf', best_rf),
        ('xgb', xgb_best)
    ],
    final_estimator=LinearRegression(),
    cv=5
)
stack_model.fit(X, y)
joblib.dump(stack_model, 'stacking_ensemble_model.pkl')

# --------------------------------------------------
# 7. Evaluation
# --------------------------------------------------
models = {
    'Decision Tree': dt,
    'Random Forest': best_rf,
    'XGBoost': xgb_best,
    'Stacking': stack_model
}

results = []
for name, model in models.items():
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring)
    results.append({
        'Model': name,
        'MAE': -scores['test_MAE'].mean(),
        'RMSE': -scores['test_RMSE'].mean(),
        'R2': scores['test_R2'].mean()
    })

df_metrics = pd.DataFrame(results).round(2)
print("\nModel Performance:\n", df_metrics)

# --------------------------------------------------
# 8. Plot with Numbers on Bars
# --------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
x = np.arange(len(df_metrics))
width = 0.6

# MAE
axs[0].bar(x, df_metrics['MAE'], color='skyblue', width=width)
axs[0].set_title('MAE')
for i, v in enumerate(df_metrics['MAE']):
    axs[0].text(i, v + 0.01, str(v), ha='center', va='bottom', fontsize=9)

# RMSE
axs[1].bar(x, df_metrics['RMSE'], color='lightgreen', width=width)
axs[1].set_title('RMSE')
for i, v in enumerate(df_metrics['RMSE']):
    axs[1].text(i, v + 0.01, str(v), ha='center', va='bottom', fontsize=9)

# R²
axs[2].bar(x, df_metrics['R2'], color='salmon', width=width)
axs[2].set_title('R²')
for i, v in enumerate(df_metrics['R2']):
    axs[2].text(i, v + 0.01, str(v), ha='center', va='bottom', fontsize=9)

# Formatting
for ax in axs:
    ax.set_xticks(x)
    ax.set_xticklabels(df_metrics['Model'], rotation=45)
    ax.margins(y=0.1)

plt.tight_layout()
plt.savefig('model_comparison.png')
plt.show()

# --------------------------------------------------
# 9. Save Results
# --------------------------------------------------
df_metrics.to_csv('model_performance_summary.csv', index=False)

print("\nSaved:")
print("- decision_tree_model.pkl")
print("- random_forest_model.pkl")
print("- xgboost_model.pkl")
print("- stacking_ensemble_model.pkl")
print("- model_performance_summary.csv")
print("- model_comparison.png")
