# STEP 1: IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# STEP 2: LOAD AND PREPROCESS DATA
column_names = [
    "crim", "zn", "indus", "chas", "nox", "rm", "age", "dis",
    "rad", "tax", "ptratio", "b", "lstat", "medv"
]
df = pd.read_csv(
    "Data Set For Task/4) house Prediction Data Set.csv",
    sep=r"\s+",
    header=None,
    names=column_names,
)
print ("data shape", df.shape)
print("missing values:", df.isnull().sum())
print ("total missing values ", df.isnull().sum().sum())
numerical_cols= df.select_dtypes(include=[np.number]).columns
for col in numerical_cols:
    if df[col].isnull().sum()>0:
        mean_value=df[col].mean()
        df[col] = df[col].fillna(mean_value)
        print(f"Filled missing values in '{col}' with mean: {mean_value:.2f}")

# STEP 3: ENCODE CATEGORICAL VARIABLES
categorical_cols=df.select_dtypes(include=['object']).columns
df_encoded=df.copy()
for col in categorical_cols:
    le= LabelEncoder()
    df_encoded[col]= le.fit_transform(df[col].astype(str))

df= df_encoded
print("all categorical variables encoded")

# STEP 4: SEPARATE FEATURES AND TARGET
if 'medv' in df.columns:
    y= df['medv']
    x=df.drop('medv', axis=1)
elif 'price' in df.columns:
    y= df['price']
    x=df.drop('price', axis=1)
elif 'Price' in df.columns:
    y= df['Price']
    x=df.drop('Price', axis=1)
else:
    x= df.iloc[:, :-1]
    y = df.iloc[:, -1]
print("X shape", x.shape)
print("X columns", x.columns)
print("Y shape", y.shape)

# STEP 5: STANDARDIZE FEATURES
scaler = StandardScaler()
x_scaled= scaler.fit_transform(x)
x_scaled = pd.DataFrame(x_scaled, columns=x.columns)
print("Features standardized")

# STEP 6: SPLIT INTO TRAIN AND TEST SETS
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)
print("Training set shape:", x_train.shape)
print("Testing set shape:", x_test.shape)

# STEP 7: CREATE AND TRAIN LINEAR REGRESSION MODEL

model=LinearRegression()
model.fit(x_train, y_train)
print("Model trained")

# STEP 8: INTERPRET MODEL COEFFICIENTS

print(f"intercept:{model.intercept_:.4f}")
coef_def= pd.DataFrame({'Features':x.columns, 'Coefficients':model.coef_}).sort_values(by='Coefficients', key=abs, ascending=False)
for idx, row in coef_def.head(10).iterrows():
    direction = "↑" if row['Coefficients'] > 0 else "↓"
    print(f"   {direction} {row['Features']:20s}: {row['Coefficients']:8.4f}")

# STEP 9: MAKE PREDICTIONS
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)
for act, pred in zip(y_test[:5], y_test_pred[:5]):
    diff = act - pred
    print(f"Actual: {act:.2f}, Predicted: {pred:.2f}, Diff: {diff:.2f}")

# STEP 10: EVALUATE MODEL PERFORMANCE
# ====== TRAINING SET EVALUATION ======
r2_train = r2_score(y_train, y_train_pred)
print(f"R² Score (Training): {r2_train:.4f}")
mse_train = mean_squared_error(y_train, y_train_pred)
print(f"Mean Squared Error (Training): {mse_train:.4f}")
rmse_train = np.sqrt(mse_train)
print(f"Root Mean Squared Error (Training): {rmse_train:.4f}")
mae_train = mean_absolute_error(y_train, y_train_pred)
print(f"Mean Absolute Error (Training): {mae_train:.4f}")
# ====== TESTING SET EVALUATION ======
r2_test = r2_score(y_test, y_test_pred)
print(f"R² Score (Testing): {r2_test:.4f}")
mse_test = mean_squared_error(y_test, y_test_pred)
print(f"Mean Squared Error (Testing): {mse_test:.4f}")
rmse_test = np.sqrt(mse_test)
print(f"Root Mean Squared Error (Testing): {rmse_test:.4f}")
mae_test = mean_absolute_error(y_test, y_test_pred)
print(f"Mean Absolute Error (Testing): {mae_test:.4f}")
# ====== OVERFITTING CHECK ======
r2_diff = r2_train - r2_test
print(f"R² Difference (Train - Test): {r2_diff:.4f}")
if r2_diff < 0.05:
    print("Model shows good generalization (no significant overfitting).")
elif r2_diff < 0.15:
    print("Model may have mild overfitting. Consider regularization or more data.")
else:
    print("Model likely overfitting. Consider regularization, more data, or simpler model.")
try:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Actual vs Predicted (Training)
    axes[0, 0].scatter(y_train, y_train_pred, alpha=0.6, color='blue')
    axes[0, 0].plot([y_train.min(), y_train.max()], 
                    [y_train.min(), y_train.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Values')
    axes[0, 0].set_ylabel('Predicted Values')
    axes[0, 0].set_title(f'Training Set: Actual vs Predicted\n(R² = {r2_train:.4f})')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Actual vs Predicted (Testing)
    axes[0, 1].scatter(y_test, y_test_pred, alpha=0.6, color='green')
    axes[0, 1].plot([y_test.min(), y_test.max()], 
                    [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 1].set_xlabel('Actual Values')
    axes[0, 1].set_ylabel('Predicted Values')
    axes[0, 1].set_title(f'Testing Set: Actual vs Predicted\n(R² = {r2_test:.4f})')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Residuals (Errors) - Training
    residuals_train = y_train - y_train_pred
    axes[1, 0].scatter(y_train_pred, residuals_train, alpha=0.6, color='blue')
    axes[1, 0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1, 0].set_xlabel('Predicted Values')
    axes[1, 0].set_ylabel('Residuals (Errors)')
    axes[1, 0].set_title('Training: Residual Plot')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Residuals (Errors) - Testing
    residuals_test = y_test - y_test_pred
    axes[1, 1].scatter(y_test_pred, residuals_test, alpha=0.6, color='green')
    axes[1, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1, 1].set_xlabel('Predicted Values')
    axes[1, 1].set_ylabel('Residuals (Errors)')
    axes[1, 1].set_title('Testing: Residual Plot')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('task2_linear_regression_results.png', dpi=300, bbox_inches='tight')
    print("✅ Visualization saved as 'task2_linear_regression_results.png'")
    plt.close()
    
except Exception as e:
    print(f"⚠️  Could not create visualization: {e}")
print("✅ Task 2 complete!")