# STEP 1: IMPORT LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score


# STEP 2: LOAD AND EXPLORE DATA
train_df = pd.read_csv("Data Set For Task/Churn Prdiction Data/churn-bigml-80.csv")
test_df = pd.read_csv("Data Set For Task/Churn Prdiction Data/churn-bigml-20.csv")

print("Training dataset shape:", train_df.shape)
print("Testing dataset shape:", test_df.shape)
print("Training columns:", list(train_df.columns))
print("Missing values in training data:")
print(train_df.isnull().sum())

target_col = "Churn"
print("Target distribution in training data:")
print(train_df[target_col].value_counts())


# STEP 3: SEPARATE FEATURES AND TARGET
X_train = train_df.drop(target_col, axis=1)
y_train = train_df[target_col]

X_test = test_df.drop(target_col, axis=1)
y_test = test_df[target_col]


# STEP 4: ENCODE CATEGORICAL FEATURES
# Random Forest needs numeric inputs, so categorical columns are converted
# into dummy variables. Test data is aligned to the same training columns.
X_train_encoded = pd.get_dummies(X_train, drop_first=False)
X_test_encoded = pd.get_dummies(X_test, drop_first=False)
X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

print("Number of features after encoding:", X_train_encoded.shape[1])


# STEP 5: TRAIN A BASELINE RANDOM FOREST MODEL
baseline_rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
)

baseline_rf.fit(X_train_encoded, y_train)
baseline_predictions = baseline_rf.predict(X_test_encoded)

print("Baseline Random Forest Results:")
print("Accuracy:", accuracy_score(y_test, baseline_predictions))
print("Precision:", precision_score(y_test, baseline_predictions))
print("Recall:", recall_score(y_test, baseline_predictions))
print("F1-score:", f1_score(y_test, baseline_predictions))


# STEP 6: TUNE HYPERPARAMETERS USING GRID SEARCH
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
}

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42, class_weight="balanced"),
    param_grid=param_grid,
    scoring="f1",
    cv=cv_strategy,
    n_jobs=-1,
)

grid_search.fit(X_train_encoded, y_train)

print("Best hyperparameters:")
print(grid_search.best_params_)
print(f"Best cross-validation F1-score: {grid_search.best_score_:.4f}")


# STEP 7: EVALUATE THE BEST MODEL WITH CROSS-VALIDATION
best_rf = grid_search.best_estimator_

cv_f1_scores = cross_val_score(
    best_rf,
    X_train_encoded,
    y_train,
    cv=cv_strategy,
    scoring="f1",
)

print("Cross-validation F1-scores:", cv_f1_scores)
print(f"Mean CV F1-score: {cv_f1_scores.mean():.4f}")
print(f"Standard deviation CV F1-score: {cv_f1_scores.std():.4f}")


# STEP 8: TEST THE FINAL MODEL
y_pred = best_rf.predict(X_test_encoded)

print("Final Random Forest Test Results:")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"F1-score : {f1_score(y_test, y_pred):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))


# STEP 9: VISUALIZE THE CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_rf.classes_)

plt.figure(figsize=(6, 5))
disp.plot(cmap="Blues", values_format="d")
plt.title("Random Forest Confusion Matrix")
plt.tight_layout()
plt.savefig("level3_task1_random_forest_confusion_matrix.png", dpi=300)
plt.show()


# STEP 10: FEATURE IMPORTANCE ANALYSIS
feature_importance = pd.DataFrame({
    "feature": X_train_encoded.columns,
    "importance": best_rf.feature_importances_,
})

feature_importance = feature_importance.sort_values(
    by="importance",
    ascending=False,
)

print("Top 15 most important features:")
print(feature_importance.head(15))

top_features = feature_importance.head(15)

plt.figure(figsize=(10, 7))
plt.barh(top_features["feature"], top_features["importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 15 Random Forest Feature Importances")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("level3_task1_random_forest_feature_importance.png", dpi=300)
plt.show()
