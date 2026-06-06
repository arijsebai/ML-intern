# STEP 1: IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix, 
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

# STEP 2: LOAD AND PREPROCESS DATA
df = pd.read_csv("Data Set For Task/1) iris.csv")
print("data set shape", df.shape)
print ("missing values:", df.isnull().sum())
numerical_cols= df.select_dtypes(include=[np.number]).columns
for col in numerical_cols:
    if df[col].isnull().sum()>0:
        mean_value=df[col].mean()
        df[col] = df[col].fillna(mean_value)

# STEP 3: ENCODE CATEGORICAL VARIABLES
categorical_cols= df.select_dtypes(include=['object']).columns
df_encoded = df.copy()
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df[col])
    label_encoders[col] = le
df=df_encoded

# STEP 4: SEPARATE FEATURES AND TARGET
x= df.iloc[:,:-1]
y = df.iloc[:,-1]
print("X shape", x.shape)
print("X columns", x.columns)
print("Y shape", y.shape)
print("unique target classes", y.unique())
for class_val in sorted (y.unique()):
    count = (y == class_val).sum()
    print(f"Class {class_val}: {count} samples")

# STEP 5: STANDARDIZE FEATURES
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
x_scaled = pd.DataFrame(x_scaled, columns=x.columns)

# STEP 6: SPLIT INTO TRAIN AND TEST SETS
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42, stratify=y)

# STEP 7: FIND THE BEST K VALUE
k_values= range(1,31)
train_accuracies =[]
test_accuracies =[]
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    train_acc = knn.score(x_train, y_train)
    test_acc = knn.score(x_test, y_test)
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)
    if k % 5 == 1 or k == 1:  # Print every 5th value
        print(f"   K={k:2d}: Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")
best_k = k_values[np.argmax(test_accuracies)]
best_test_acc = max(test_accuracies)
print(f"Best K value: {best_k}")
print(f"Best test accuracy: {best_test_acc:.4f} ({best_test_acc*100:.2f}%)")

# STEP 8: TRAIN FINAL KNN MODEL WITH BEST K
knn_model= KNeighborsClassifier(n_neighbors=best_k)
knn_model.fit(x_train, y_train)

# STEP 9: MAKE PREDICTIONS
y_train_pred = knn_model.predict(x_train)
y_test_pred = knn_model.predict(x_test)
y_test_proba = knn_model.predict_proba(x_test)

# STEP 10: EVALUATE MODEL PERFORMANCE
accuracy_train = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {accuracy_train:.4f} ({accuracy_train*100:.2f}%)")
accuracy_test = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy: {accuracy_test:.4f} ({accuracy_test*100:.2f}%)")
precision = precision_score(y_test, y_test_pred, average='weighted')
recall = recall_score(y_test, y_test_pred, average='weighted')
f1 = f1_score(y_test, y_test_pred, average='weighted')
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
cm= confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", classification_report(y_test, y_test_pred, target_names=[f"Class {i}" for i in sorted(y.unique())]))
# ====== OVERFITTING CHECK ======
acc_diff = abs(accuracy_train - accuracy_test)
print(f"Accuracy Difference (Train - Test): {acc_diff:.4f}")
if acc_diff < 0.05:
    print("Good! Model generalizes well (no overfitting)")
elif acc_diff < 0.15:
    print("Slight overfitting detected, but acceptable")
else:
    print("Significant overfitting - model learned training data too well")

# STEP 11: VISUALIZE RESULTS
try:
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: K Value Performance
    axes[0, 0].plot(k_values, train_accuracies, 'b-o', label='Training Accuracy', linewidth=2)
    axes[0, 0].plot(k_values, test_accuracies, 'r-s', label='Testing Accuracy', linewidth=2)
    axes[0, 0].axvline(x=best_k, color='green', linestyle='--', linewidth=2, label=f'Best K={best_k}')
    axes[0, 0].set_xlabel('K Value', fontsize=11)
    axes[0, 0].set_ylabel('Accuracy', fontsize=11)
    axes[0, 0].set_title('KNN Performance: Finding Best K Value', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Confusion Matrix
    im = axes[0, 1].imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=axes[0, 1], label='Count')
    class_labels = [f"Class {c}" for c in sorted(y_test.unique())]
    axes[0, 1].set_xticks(range(len(class_labels)))
    axes[0, 1].set_yticks(range(len(class_labels)))
    axes[0, 1].set_xticklabels(class_labels)
    axes[0, 1].set_yticklabels(class_labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[0, 1].text(j, i, cm[i, j], ha='center', va='center', color='black', fontweight='bold')
    axes[0, 1].set_xlabel('Predicted', fontsize=11)
    axes[0, 1].set_ylabel('Actual', fontsize=11)
    axes[0, 1].set_title('Confusion Matrix (Test Set)', fontsize=12, fontweight='bold')
    
    # Plot 3: Accuracy Comparison
    categories = ['Training', 'Testing']
    accuracies = [accuracy_train, accuracy_test]
    colors = ['#3498db', '#e74c3c']
    bars = axes[1, 0].bar(categories, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    axes[1, 0].set_ylabel('Accuracy', fontsize=11)
    axes[1, 0].set_title('Model Accuracy Comparison', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{acc:.2%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Plot 4: Precision, Recall, F1 Score
    metrics_names = ['Precision', 'Recall', 'F1-Score']
    metrics_values = [precision, recall, f1]
    colors = ['#2ecc71', '#f39c12', '#9b59b6']
    bars = axes[1, 1].bar(metrics_names, metrics_values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    axes[1, 1].set_ylabel('Score', fontsize=11)
    axes[1, 1].set_title('Classification Metrics', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, metrics_values):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('task3_knn_results.png', dpi=300, bbox_inches='tight')
    print("✅ Visualization saved as 'task3_knn_results.png'")
    plt.close()
    
except Exception as e:
    print(f"⚠️  Could not create visualization: {e}")