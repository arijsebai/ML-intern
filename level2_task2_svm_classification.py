# STEP 1: IMPORT LIBRARIES
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score,recall_score,f1_score, confusion_matrix,classification_report, roc_curve, auc, roc_auc_score)
import seaborn as sns
from sklearn.decomposition import PCA

# STEP 2: LOAD AND EXPLORE DATA
df = pd.read_csv("Data Set For Task/Churn Prdiction Data/churn-bigml-80.csv")
print ("first few rows", df.head())
print ("Data types", df.dtypes)
missing_count = df.isnull().sum()
if missing_count.sum() > 0:
    print("Missing values detected:")
    print(missing_count[missing_count > 0])
else:
    print("No missing values detected.")

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
print(f"Target Column: {target_col}")
print("Target Variable Distribution:")
churn_counts = df[target_col].value_counts()
for class_val in sorted(df[target_col].unique()):
    count = (df[target_col] == class_val).sum()
    pct = (count / len(df)) * 100
    print(f"   {class_val}: {count} ({pct:.1f}%)")
imbalance_ratio = max(churn_counts) / min(churn_counts)
if imbalance_ratio > 3:
    print("significant class imbalance detected")
else:
    print("class distribution is relatively balanced")

# STEP 3: DATA PREPROCESSING
x = df.drop(target_col, axis=1)
y = df[target_col]
if y.dtype == 'object':
    le_target = LabelEncoder()
    y = le_target.fit_transform(y)
    print(f"Target classes encoded: {dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))}")

# STEP 4: ENCODE CATEGORICAL VARIABLES
categorical_cols = x.select_dtypes(include=['object']).columns
x_encoded = x.copy()
label_encoded = {}
for col in categorical_cols:
    le = LabelEncoder()
    x_encoded[col]= le.fit_transform(x[col].astype(str))
    label_encoded[col]= le
    print(f"Encoded '{col}' ({len(le.classes_)} categories)")

x= x_encoded

# STEP 5: SPLIT INTO TRAIN AND TEST SETS
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 42 , stratify = y)

# STEP 6: STANDARDIZE FEATURES
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
x_train_scaled = pd.DataFrame(x_train_scaled, columns = x.columns)
x_test_scaled = pd.DataFrame(x_test_scaled, columns = x.columns)

# STEP 7: TRAIN SVM WITH DIFFERENT KERNELS
models ={}
results = {}
# ====== LINEAR KERNEL ======
svm_linear = SVC(kernel= "linear", C=1.0, random_state=42, probability=True)
svm_linear.fit(x_train_scaled, y_train)
y_train_pred_linear = svm_linear.predict(x_train_scaled)
y_test_pred_linear = svm_linear.predict(x_test_scaled)
y_test_pred_proba_linear = svm_linear.predict_proba(x_test_scaled)
models['linear']= svm_linear
results['linear'] ={
    'y_test_pred': y_test_pred_linear,
    'y_test_pred_proba': y_test_pred_proba_linear,
    'train_acc' : accuracy_score(y_train, y_train_pred_linear),
    'test_acc': accuracy_score(y_test, y_test_pred_linear),
}
print(f"Linear kernel trained!")
print(f"   Training Accuracy: {results['linear']['train_acc']:.4f}")
print(f"   Testing Accuracy: {results['linear']['test_acc']:.4f}")
print(f"   Support Vectors: {len(svm_linear.support_)} out of {len(x_train)}")
# ====== RBF KERNEL ======
svm_rbf = SVC(kernel="rbf", C= 1.0, random_state=42, probability = True )
svm_rbf.fit(x_train_scaled, y_train)
y_train_pred_rbf = svm_rbf.predict(x_train_scaled)
y_test_pred_rbf = svm_rbf.predict(x_test_scaled)
y_test_pred_proba_rbf = svm_rbf.predict_proba(x_test_scaled)
models['rbf']= svm_rbf
results['rbf'] ={
    'y_test_pred': y_test_pred_rbf,
    'y_test_pred_proba': y_test_pred_proba_rbf,
    'train_acc' : accuracy_score(y_train, y_train_pred_rbf),
    'test_acc': accuracy_score(y_test, y_test_pred_rbf),
}
print(f"RBF kernel trained!")
print(f"   Training Accuracy: {results['rbf']['train_acc']:.4f}")
print(f"   Testing Accuracy: {results['rbf']['test_acc']:.4f}")
print(f"   Support Vectors: {len(svm_rbf.support_)} out of {len(x_train)}")
# ====== POLYNOMIAL KERNEL ======
svm_poly = SVC(kernel="poly", degree=3, C= 1.0, random_state=42, probability = True)
svm_poly.fit(x_train_scaled, y_train)
y_train_pred_poly = svm_poly.predict(x_train_scaled)
y_test_pred_poly = svm_poly.predict(x_test_scaled)
y_test_pred_proba_poly = svm_poly.predict_proba(x_test_scaled)
models['poly'] = svm_poly
results['poly'] ={
    'y_test_pred': y_test_pred_poly,
    'y_test_pred_proba': y_test_pred_proba_poly,
    'train_acc' : accuracy_score(y_train, y_train_pred_poly),
    'test_acc': accuracy_score(y_test, y_test_pred_poly),
}

# STEP 8: COMPARE KERNELS
print("Kernel Comparison Summary:")
for kernel in ['linear', 'rbf', 'poly']:
    train_acc = results[kernel]['train_acc']
    test_acc = results[kernel]['test_acc']
    n_support = len(models[kernel].support_)
    print(f"{kernel:<15} {train_acc:>14.4f} {test_acc:>14.4f} {n_support:>19}")
best_kernel = max(results, key=lambda k: results[k]['test_acc'])
print(f"Best performing kernel: {best_kernel}")

# STEP 9: DETAILED EVALUATION OF BEST MODEL
best_model = models[best_kernel]
y_test_pred_best = results[best_kernel]['y_test_pred']
y_test_pred_proba_best = results[best_kernel]['y_test_pred_proba']
# ====== TRAINING SET EVALUATION ======
acc_train = results[best_kernel]['train_acc']
precision_train= precision_score(y_train, best_model.predict(x_train_scaled))
recall_train = recall_score(y_train, best_model.predict(x_train_scaled))
f1_train = f1_score(y_train, best_model.predict(x_train_scaled))
print(f"\nBest Model ({best_kernel} kernel) Training Performance:")
print(f"   Accuracy: {acc_train:.4f} ({acc_train*100:.2f}%)")
print(f"   Precision: {precision_train:.4f}")
print(f"   Recall: {recall_train:.4f}")
print(f"   F1 Score: {f1_train:.4f}")
# ====== TESTING SET EVALUATION ======
accuracy_test = accuracy_score(y_test, y_test_pred_best)
precision_test = precision_score(y_test, y_test_pred_best)
recall_test = recall_score(y_test, y_test_pred_best)
f1_test = f1_score(y_test, y_test_pred_best)
auc_score = roc_auc_score(y_test, y_test_pred_proba_best[:, 1])
print(f"\nBest Model ({best_kernel} kernel) Test Performance:")
print(f"   Accuracy: {accuracy_test:.4f} ({accuracy_test*100:.2f}%)")
print(f"   Precision: {precision_test:.4f}")
print(f"   Recall: {recall_test:.4f}")
print(f"   F1 Score: {f1_test:.4f}")
print(f"   AUC Score: {auc_score:.4f}")
# ====== CONFUSION MATRIX ======
cm = confusion_matrix(y_test, y_test_pred_best)
print(f"\n{cm}")
tn, fp, fn, tp = cm.ravel()
print(f"\n   True Negatives (TN):  {tn}")
print(f"   False Positives (FP): {fp}")
print(f"   False Negatives (FN): {fn}")
print(f"   True Positives (TP):  {tp}")
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
print(f"\n   Sensitivity (Recall): {sensitivity:.4f}")
print(f"   Specificity: {specificity:.4f}")
# ====== CROSS-VALIDATION ======
cv_scores = cross_val_score(best_model, x_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"\nCross-Validation Scores: {cv_scores}")
print(f"   Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
# STEP 10: ROC CURVE ANALYSIS
fpr, tpr, thresholds = roc_curve(y_test, y_test_pred_proba_best[:, 1])
print(f"\n ROC Curve Metrics:")
print(f"   AUC Score: {auc_score:.4f}")
if auc_score >= 0.9:
    rating = "Excellent!"
elif auc_score >= 0.8:
    rating = "Very Good"
elif auc_score >= 0.7:
    rating = "Good"
else:
    rating = "Needs improvement"
print(f"   Rating: {rating}")

# STEP 11: VISUALIZE DECISION BOUNDARY (2D)
pca = PCA(n_components=2)
X_train_2d = pca.fit_transform(x_train_scaled)
X_test_2d = pca.transform(x_test_scaled)
svm_2d = SVC(kernel=best_kernel, C=1.0, random_state=42)
svm_2d.fit(X_train_2d, y_train)
print(f"   Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
print(f"   2D model trained for visualization")

# STEP 12: CREATE VISUALIZATIONS
try:
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Kernel Comparison
    kernels = list(results.keys())
    train_accuracies = [results[k]['train_acc'] for k in kernels]
    test_accuracies = [results[k]['test_acc'] for k in kernels]
    
    x = np.arange(len(kernels))
    width = 0.35
    
    bars1 = axes[0, 0].bar(x - width/2, train_accuracies, width, label='Training', 
                           alpha=0.8, color='#3498db', edgecolor='black')
    bars2 = axes[0, 0].bar(x + width/2, test_accuracies, width, label='Testing', 
                           alpha=0.8, color='#e74c3c', edgecolor='black')
    
    axes[0, 0].set_ylabel('Accuracy', fontsize=11)
    axes[0, 0].set_title('SVM Kernel Comparison', fontsize=12, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(kernels)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].set_ylim([0, 1.1])
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1],
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'],
                cbar_kws={'label': 'Count'})
    axes[0, 1].set_xlabel('Predicted', fontsize=11)
    axes[0, 1].set_ylabel('Actual', fontsize=11)
    axes[0, 1].set_title(f'Confusion Matrix ({best_kernel.upper()} Kernel)', fontsize=12, fontweight='bold')
    
    # Plot 3: ROC Curve
    axes[1, 0].plot(fpr, tpr, color='#e74c3c', lw=2, label=f'ROC Curve (AUC = {auc_score:.3f})')
    axes[1, 0].plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Classifier')
    axes[1, 0].set_xlabel('False Positive Rate', fontsize=11)
    axes[1, 0].set_ylabel('True Positive Rate', fontsize=11)
    axes[1, 0].set_title('ROC Curve', fontsize=12, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Decision Boundary (2D)
    h = 0.02  # Step size in mesh
    
    x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
    y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    axes[1, 1].contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    axes[1, 1].contour(xx, yy, Z, colors='black', linewidths=0.5, levels=[0.5])
    
    # Plot training points
    scatter = axes[1, 1].scatter(X_train_2d[:, 0], X_train_2d[:, 1], 
                                 c=y_train, cmap='coolwarm', 
                                 edgecolors='black', alpha=0.6, s=50)
    
    # Highlight support vectors
    support_vectors = svm_2d.support_
    axes[1, 1].scatter(X_train_2d[support_vectors, 0], X_train_2d[support_vectors, 1],
                       s=200, linewidth=1.5, facecolors='none', edgecolors='yellow',
                       label=f'Support Vectors ({len(support_vectors)})')
    
    axes[1, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=11)
    axes[1, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=11)
    axes[1, 1].set_title(f'Decision Boundary ({best_kernel.upper()} Kernel, 2D PCA)', 
                        fontsize=12, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig('level2_task2_svm_results.png', dpi=300, bbox_inches='tight')
    print("✅ Visualization saved as 'level2_task2_svm_results.png'")
    plt.close()
    
except Exception as e:
    print(f"⚠️  Could not create visualization: {e}")