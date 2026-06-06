# STEP 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

# STEP 2: Load & Explore the Dataset
df=pd.read_csv("Data Set For Task/1) iris.csv")
print("Data set shape", df.shape)
print("Data set columns names and data types:", df.dtypes)
print("Missing values:", df.isnull().sum())

# STEP 3: Split Data
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
feature_names = list(X.columns)
class_names = sorted(y.unique())
print("Feature columns:", feature_names)
print("Target classes:", class_names)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training samples : {X_train.shape[0]}")
print(f"Testing samples  : {X_test.shape[0]}")

# STEP 4: Train FULL Tree (no pruning)
dt_full = DecisionTreeClassifier(random_state=42)
dt_full.fit(X_train, y_train)
print(f"Tree depth       : {dt_full.get_depth()}")
print(f"Number of leaves : {dt_full.get_n_leaves()}")
y_pred_full = dt_full.predict(X_test)
acc_full = accuracy_score(y_test, y_pred_full)
f1_full  = f1_score(y_test, y_pred_full, average="weighted")
print(f"Accuracy         : {acc_full:.4f} ({acc_full*100:.2f}%)")
print(f"F1-Score (wt.)   : {f1_full:.4f}")

# STEP 5: Prune Tree (max_depth)
dt_pruned = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_pruned.fit(X_train, y_train)
print(f"Tree depth       : {dt_pruned.get_depth()}")
print(f"Number of leaves : {dt_pruned.get_n_leaves()}")
y_pred_pruned = dt_pruned.predict(X_test)
acc_pruned = accuracy_score(y_test, y_pred_pruned)
f1_pruned  = f1_score(y_test, y_pred_pruned, average="weighted")
print(f"Accuracy         : {acc_pruned:.4f} ({acc_pruned*100:.2f}%)")
print(f"F1-Score (wt.)   : {f1_pruned:.4f}")

# STEP 6: Evaluation Report
print(classification_report(y_test, y_pred_pruned, target_names=class_names))

# STEP 7: Text representation of tree
tree_rules = export_text(dt_pruned, feature_names=feature_names)
print(tree_rules)

# STEP 8: Create Beautiful Visualizations
COLORS = {
    "bg"       : "#0d1117",
    "panel"    : "#161b22",
    "border"   : "#30363d",
    "accent1"  : "#58a6ff",   # blue
    "accent2"  : "#3fb950",   # green
    "accent3"  : "#f78166",   # red/orange
    "accent4"  : "#d2a8ff",   # purple
    "text"     : "#e6edf3",
    "subtext"  : "#8b949e",
    "setosa"   : "#79c0ff",
    "versicolor": "#56d364",
    "virginica" : "#ff7b72",
    "node_colors": ["#79c0ff", "#56d364", "#ff7b72"],
}

plt.rcParams.update({
    "figure.facecolor"  : COLORS["bg"],
    "axes.facecolor"    : COLORS["panel"],
    "axes.edgecolor"    : COLORS["border"],
    "axes.labelcolor"   : COLORS["text"],
    "xtick.color"       : COLORS["subtext"],
    "ytick.color"       : COLORS["subtext"],
    "text.color"        : COLORS["text"],
    "grid.color"        : COLORS["border"],
    "grid.linestyle"    : "--",
    "grid.alpha"        : 0.5,
    "font.family"       : "monospace",
})

# ── Figure 1: Full Tree ──────────────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(22, 10), facecolor=COLORS["bg"])
fig1.suptitle("Full Decision Tree (No Pruning) — Iris Dataset",
              fontsize=16, color=COLORS["accent1"], fontweight="bold", y=0.98)

plot_tree(
    dt_full,
    feature_names=feature_names,
    class_names=class_names,
    filled=True,
    rounded=True,
    ax=ax1,
    fontsize=8,
    impurity=True,
    proportion=False,
)
ax1.set_facecolor(COLORS["bg"])

info_text = (
    f"  Depth: {dt_full.get_depth()}   |   "
    f"Leaves: {dt_full.get_n_leaves()}   |   "
    f"Accuracy: {acc_full*100:.1f}%   |   "
    f"F1: {f1_full:.3f}  "
)
fig1.text(0.5, 0.01, info_text, ha="center", fontsize=11,
          color=COLORS["subtext"],
          bbox=dict(boxstyle="round,pad=0.4", facecolor=COLORS["panel"],
                    edgecolor=COLORS["border"]))

plt.tight_layout(rect=[0, 0.04, 1, 0.96])
fig1.savefig("level2_task2_tree_full.png", dpi=150,
             bbox_inches="tight", facecolor=COLORS["bg"])
print("  ✅ tree_full.png saved")

# ── Figure 2: Pruned Tree ────────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(18, 8), facecolor=COLORS["bg"])
fig2.suptitle("Pruned Decision Tree (max_depth=3) — Iris Dataset",
              fontsize=16, color=COLORS["accent2"], fontweight="bold", y=0.98)

plot_tree(
    dt_pruned,
    feature_names=feature_names,
    class_names=class_names,
    filled=True,
    rounded=True,
    ax=ax2,
    fontsize=11,
    impurity=True,
    proportion=False,
)
ax2.set_facecolor(COLORS["bg"])

info_text2 = (
    f"  Depth: {dt_pruned.get_depth()}   |   "
    f"Leaves: {dt_pruned.get_n_leaves()}   |   "
    f"Accuracy: {acc_pruned*100:.1f}%   |   "
    f"F1: {f1_pruned:.3f}  "
)
fig2.text(0.5, 0.01, info_text2, ha="center", fontsize=11,
          color=COLORS["subtext"],
          bbox=dict(boxstyle="round,pad=0.4", facecolor=COLORS["panel"],
                    edgecolor=COLORS["border"]))

plt.tight_layout(rect=[0, 0.04, 1, 0.96])
fig2.savefig("level2_task2_tree_pruned.png", dpi=150,
             bbox_inches="tight", facecolor=COLORS["bg"])
print("  ✅ tree_pruned.png saved")

# ── Figure 3: Dashboard (Confusion + Metrics + Feature Importance) ───────────
fig3 = plt.figure(figsize=(18, 12), facecolor=COLORS["bg"])
fig3.suptitle("Model Evaluation Dashboard — Iris Decision Tree",
              fontsize=18, color=COLORS["text"], fontweight="bold", y=0.97)

gs = gridspec.GridSpec(2, 3, figure=fig3,
                       hspace=0.45, wspace=0.35,
                       top=0.90, bottom=0.07,
                       left=0.07, right=0.96)

# ── 3a: Confusion Matrix (Full) ──────────────────
ax_cm1 = fig3.add_subplot(gs[0, 0])
cm_full = confusion_matrix(y_test, y_pred_full)
disp1 = ConfusionMatrixDisplay(cm_full, display_labels=class_names)
disp1.plot(ax=ax_cm1, colorbar=False, cmap="Blues")
ax_cm1.set_title(f"Confusion Matrix — Full Tree\nAcc: {acc_full*100:.1f}%",
                 color=COLORS["accent1"], fontsize=11, pad=8)
ax_cm1.tick_params(axis="x", rotation=30)
for t in ax_cm1.texts:
    t.set_color("white")
    t.set_fontsize(13)

# ── 3b: Confusion Matrix (Pruned) ────────────────
ax_cm2 = fig3.add_subplot(gs[0, 1])
cm_pruned = confusion_matrix(y_test, y_pred_pruned)
disp2 = ConfusionMatrixDisplay(cm_pruned, display_labels=class_names)
disp2.plot(ax=ax_cm2, colorbar=False, cmap="Greens")
ax_cm2.set_title(f"Confusion Matrix — Pruned Tree\nAcc: {acc_pruned*100:.1f}%",
                 color=COLORS["accent2"], fontsize=11, pad=8)
ax_cm2.tick_params(axis="x", rotation=30)
for t in ax_cm2.texts:
    t.set_color("white")
    t.set_fontsize(13)

# ── 3c: Feature Importance ───────────────────────
ax_fi = fig3.add_subplot(gs[0, 2])
importances = dt_pruned.feature_importances_
feat_names  = [n.replace(" (cm)", "").replace("_", "\n").replace(" ", "\n") for n in feature_names]
bar_colors  = [COLORS["accent1"], COLORS["accent2"], COLORS["accent3"], COLORS["accent4"]]
bars = ax_fi.barh(feat_names, importances, color=bar_colors,
                  edgecolor=COLORS["border"], linewidth=0.8, height=0.55)
ax_fi.set_xlabel("Importance", color=COLORS["subtext"])
ax_fi.set_title("Feature Importance\n(Pruned Tree)", color=COLORS["accent4"], fontsize=11)
ax_fi.set_xlim(0, max(importances) * 1.25)
for bar, val in zip(bars, importances):
    ax_fi.text(val + 0.01, bar.get_y() + bar.get_height()/2,
               f"{val:.3f}", va="center", fontsize=10, color=COLORS["text"])

# ── 3d: Accuracy vs Depth ────────────────────────
ax_depth = fig3.add_subplot(gs[1, 0:2])
depths      = range(1, 10)
train_accs  = []
test_accs   = []
for d in depths:
    clf = DecisionTreeClassifier(max_depth=d, random_state=42)
    clf.fit(X_train, y_train)
    train_accs.append(accuracy_score(y_train, clf.predict(X_train)))
    test_accs.append(accuracy_score(y_test,  clf.predict(X_test)))

ax_depth.plot(depths, train_accs, "o-", color=COLORS["accent1"],
              lw=2.5, ms=7, label="Train Accuracy")
ax_depth.plot(depths, test_accs,  "s-", color=COLORS["accent2"],
              lw=2.5, ms=7, label="Test Accuracy")
ax_depth.axvline(x=3, color=COLORS["accent3"], ls="--", lw=1.8,
                 label="Chosen depth (3)")
ax_depth.fill_between(depths, train_accs, test_accs,
                      alpha=0.12, color=COLORS["accent3"],
                      label="Overfitting gap")
ax_depth.set_xlabel("Tree Depth")
ax_depth.set_ylabel("Accuracy")
ax_depth.set_title("Accuracy vs Tree Depth — Overfitting Analysis",
                   color=COLORS["accent1"], fontsize=11)
ax_depth.legend(facecolor=COLORS["panel"], edgecolor=COLORS["border"],
                labelcolor=COLORS["text"], fontsize=9)
ax_depth.set_xticks(list(depths))
ax_depth.grid(True)

# ── 3e: Per-class F1 comparison ──────────────────
ax_f1 = fig3.add_subplot(gs[1, 2])
classes = class_names
f1_full_per   = f1_score(y_test, y_pred_full,   average=None)
f1_pruned_per = f1_score(y_test, y_pred_pruned, average=None)
x = np.arange(len(classes))
w = 0.35
ax_f1.bar(x - w/2, f1_full_per,   w, label="Full",   color=COLORS["accent1"],
          edgecolor=COLORS["border"])
ax_f1.bar(x + w/2, f1_pruned_per, w, label="Pruned", color=COLORS["accent2"],
          edgecolor=COLORS["border"])
ax_f1.set_xticks(x)
ax_f1.set_xticklabels(classes, fontsize=9)
ax_f1.set_ylabel("F1-Score")
ax_f1.set_ylim(0, 1.15)
ax_f1.set_title("Per-class F1-Score\nFull vs Pruned",
                color=COLORS["accent2"], fontsize=11)
ax_f1.legend(facecolor=COLORS["panel"], edgecolor=COLORS["border"],
             labelcolor=COLORS["text"], fontsize=9)
ax_f1.grid(True, axis="y")
for bar in ax_f1.patches:
    h = bar.get_height()
    ax_f1.text(bar.get_x() + bar.get_width()/2, h + 0.02,
               f"{h:.2f}", ha="center", va="bottom", fontsize=8,
               color=COLORS["text"])

fig3.savefig("level2_task2_evaluation_dashboard.png", dpi=150,
             bbox_inches="tight", facecolor=COLORS["bg"])
print("  ✅ evaluation_dashboard.png saved")