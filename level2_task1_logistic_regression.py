# STEP 1: IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression  # The algorithm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    roc_auc_score,
    log_loss
)
import seaborn as sns
# STEP 2: LOAD AND EXPLORE DATA
df = pd.read_csv("Data Set For Task/Churn Prdiction Data/churn-bigml-80.csv")
if 'Churn' in df.columns:
    target_col = 'Churn'
elif 'churn' in df.columns:
    target_col = 'churn'
else:
    # Find binary column
    target_col = None
    for col in df.columns:
        if df[col].nunique() == 2:
            target_col = col
            break
print("Target Variable Distribution:")
churn_counts = df[target_col].value_counts()
for class_val in sorted(df[target_col].unique()):
    count = (df[target_col] == class_val).sum()
    pct = (count / len(df)) * 100
    print(f"   {class_val}: {count} ({pct:.1f}%)")
imbalance_ratio = max(churn_counts) / min(churn_counts)
if imbalance_ratio > 3:
    print(f"Class Imbalance Detected! Ratio: {imbalance_ratio:.2f}")
    print(f"   (This affects model performance - using class_weight='balanced')")
else:
    print(f"Class distribution is balanced (ratio: {imbalance_ratio:.2f})")

# STEP 3: DATA PREPROCESSING
X = df.drop(target_col, axis=1)
y = df[target_col]
if y.dtype == 'object':
    le_target = LabelEncoder()
    y = le_target.fit_transform(y)
    print(f"\n🔤 Target classes encoded: {dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))}")

# STEP 4: ENCODE CATEGORICAL VARIABLES
categorical_cols = X.select_dtypes(include=['object']).columns
X_encoded = X.copy()
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le
    print(f"\n   ✅ Encoded '{col}' ({len(le.classes_)} categories)")
    if len(le.classes_) <= 5:
        for i, class_val in enumerate(le.classes_):
            print(f"      {class_val} → {i}")

X = X_encoded

# STEP 5: SPLIT INTO TRAIN AND TEST SETS
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# STEP 6: STANDARDIZE FEATURES
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
x_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
x_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
print("Features standardized")

# STEP 7: TRAIN LOGISTIC REGRESSION MODEL
model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
model.fit(x_train_scaled, y_train)
print("Model trained")
y_train_pred = model.predict(x_train_scaled)
y_test_pred = model.predict(x_test_scaled)
y_train_pred_proba = model.predict_proba(x_train_scaled)[:, 1]
y_test_pred_proba = model.predict_proba(x_test_scaled)[:, 1]

# STEP 8: INTERPRET MODEL COEFFICIENTS
intercept = model.intercept_[0]
coefficients = model.coef_[0]
print(f"\nModel Intercept: {intercept:.4f}")
print("\nFeature Coefficients:")
coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': coefficients, 'Odds_Ratio': np.exp(coefficients)}).sort_values(by='Odds_Ratio', ascending=False)
for index, row in coef_df.iterrows():
    direction = "↑" if row['Coefficient'] > 0 else "↓"
    print(f"   {direction} {row['Feature']:20s}: Coef={row['Coefficient']:.4f}, Odds Ratio={row['Odds_Ratio']:.4f}")

# STEP 9: UNDERSTAND PROBABILITY PREDICTIONS
for i in range(min(15, len(y_test))):
    actual = y_test.iloc[i]
    predicted = y_test_pred[i]
    prob_0 = y_test_pred_proba[i]
    prob_1 = y_test_pred_proba[i]
    correct = "✅" if actual == predicted else "❌"
    
    print(f"{i+1:<4} {actual:<10} {predicted:<10} {prob_0:<12.4f} {prob_1:<12.4f} {correct:<8}")

# STEP 10: EVALUATE MODEL PERFORMANCE
accuracy_train = accuracy_score(y_train, y_train_pred)
precision_train = precision_score(y_train, y_train_pred)
recall_train = recall_score(y_train, y_train_pred)
f1_train = f1_score(y_train, y_train_pred)
print(f"\nTraining Set Performance:")
print(f"   Accuracy: {accuracy_train:.4f} ({accuracy_train*100:.2f}%)")
print(f"   Precision: {precision_train:.4f}")
print(f"   Recall: {recall_train:.4f}")
print(f"   F1 Score: {f1_train:.4f}")

accuracy_test = accuracy_score(y_test, y_test_pred)
precision_test = precision_score(y_test, y_test_pred)
recall_test = recall_score(y_test, y_test_pred)
f1_test = f1_score(y_test, y_test_pred)
auc_score = roc_auc_score(y_test, y_test_pred_proba)
print(f"\nTest Set Performance:")
print(f"   Accuracy: {accuracy_test:.4f} ({accuracy_test*100:.2f}%)")
print(f"   Precision: {precision_test:.4f}")
print(f"   Recall: {recall_test:.4f}")
print(f"   F1 Score: {f1_test:.4f}")
print(f"   AUC Score: {auc_score:.4f}")
cm = confusion_matrix(y_test, y_test_pred)
print(f"\n{cm}")

# STEP 11: ROC CURVE ANALYSIS
fpr, tpr, thresholds = roc_curve(y_test, y_test_pred_proba)
if auc_score >= 0.9:
    rating = "Excellent!"
elif auc_score >= 0.8:
    rating = "Very Good"
elif auc_score >= 0.7:
    rating = "Good 👍"
else:
    rating = "Needs improvement"
print(f"\nROC AUC Score: {auc_score:.4f} → {rating}")

# STEP 12: VISUALIZE RESULTS
try:
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Confusion Matrix Heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'],
                cbar_kws={'label': 'Count'})
    axes[0, 0].set_xlabel('Predicted', fontsize=11)
    axes[0, 0].set_ylabel('Actual', fontsize=11)
    axes[0, 0].set_title('Confusion Matrix (Test Set)', fontsize=12, fontweight='bold')
    
    # Plot 2: ROC Curve
    axes[0, 1].plot(fpr, tpr, color='#e74c3c', lw=2, label=f'ROC Curve (AUC = {auc_score:.3f})')
    axes[0, 1].plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Classifier')
    axes[0, 1].set_xlabel('False Positive Rate', fontsize=11)
    axes[0, 1].set_ylabel('True Positive Rate', fontsize=11)
    axes[0, 1].set_title('ROC Curve', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Model Performance Metrics
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    metrics_values_train = [accuracy_train, precision_train, recall_train, f1_train]
    metrics_values_test = [accuracy_test, precision_test, recall_test, f1_test]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    bars1 = axes[1, 0].bar(x - width/2, metrics_values_train, width, label='Training', alpha=0.8, color='#3498db')
    bars2 = axes[1, 0].bar(x + width/2, metrics_values_test, width, label='Testing', alpha=0.8, color='#e74c3c')
    
    axes[1, 0].set_ylabel('Score', fontsize=11)
    axes[1, 0].set_title('Model Performance Comparison', fontsize=12, fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(metrics_names)
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].set_ylim([0, 1.1])
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Top Feature Importance (by coefficient magnitude)
    top_features = coef_df.head(10)
    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in top_features['Coefficient']]
    
    axes[1, 1].barh(range(len(top_features)), top_features['Coefficient'], color=colors, alpha=0.7, edgecolor='black')
    axes[1, 1].set_yticks(range(len(top_features)))
    axes[1, 1].set_yticklabels(top_features['Feature'], fontsize=10)
    axes[1, 1].set_xlabel('Coefficient', fontsize=11)
    axes[1, 1].set_title('Top 10 Features by Impact', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    axes[1, 1].axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    
    plt.tight_layout()
    plt.savefig('level2_task1_logistic_regression_results.png', dpi=300, bbox_inches='tight')
    print("✅ Visualization saved as 'level2_task1_logistic_regression_results.png'")
    plt.close()
    
except Exception as e:
    print(f"⚠️  Could not create visualization: {e}")

# STEP 13: BUSINESS INSIGHTS
top_churn_features = coef_df[coef_df['Coefficient'] > 0].head(3)
top_retention_features = coef_df[coef_df['Coefficient'] < 0].head(3)

print(f"\n🔴 TOP CHURN RISK FACTORS (increase churn probability):")
for idx, (i, row) in enumerate(top_churn_features.iterrows(), 1):
    print(f"   {idx}. {row['Feature']}: {row['Odds_Ratio']:.2f}x odds increase")
    print(f"      ({(row['Odds_Ratio']-1)*100:.1f}% increase in churn odds)")

print(f"\n🟢 TOP RETENTION FACTORS (decrease churn probability):")
for idx, (i, row) in enumerate(top_retention_features.iterrows(), 1):
    print(f"   {idx}. {row['Feature']}: {row['Odds_Ratio']:.2f}x odds reduction")
    print(f"      ({(1-row['Odds_Ratio'])*100:.1f}% reduction in churn odds)")
print("✅ Level 2 - Task 1 complete!")