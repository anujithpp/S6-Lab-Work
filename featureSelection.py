import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend, no display needed
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2, f_classif, RFE
from sklearn.metrics import accuracy_score, classification_report

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# =========================================================
# a. Load Dataset and Split into Features (X) and Labels (y)
# =========================================================
print("=" * 60)
print("LOADING IRIS DATASET")
print("=" * 60)

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='target')
target_names = iris.target_names
feature_names = iris.feature_names

print(f"\nDataset Shape: {X.shape}")
print(f"Features: {list(feature_names)}")
print(f"Classes: {list(target_names)}")
print(f"\nFirst 5 rows:\n{X.head()}")
print(f"\nClass Distribution:\n{y.value_counts().rename(index=dict(enumerate(target_names)))}")

# =========================================================
# b. Exploratory Data Analysis (EDA)
# =========================================================
print("\n" + "=" * 60)
print("EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 60)

print("\nStatistical Summary:")
print(X.describe().round(3))

print(f"\nMissing Values: {X.isnull().sum().sum()}")

# EDA Visualizations
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Distribution of each feature
for i, feature in enumerate(feature_names):
    ax = axes[0, i] if i < 3 else axes[1, i - 3]
    for cls_idx, cls_name in enumerate(target_names):
        ax.hist(X[feature][y == cls_idx], alpha=0.6, label=cls_name, bins=15)
    ax.set_title(f'Distribution: {feature}', fontsize=12, fontweight='bold')
    ax.set_xlabel(feature)
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)

# 2. Correlation Heatmap
ax = axes[1, 0]
corr = X.corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax,
            xticklabels=[f.split(' ')[0] for f in feature_names],
            yticklabels=[f.split(' ')[0] for f in feature_names])
ax.set_title('Feature Correlation Heatmap', fontsize=12, fontweight='bold')

# 3. Box plots
ax = axes[1, 1]
X_melted = X.copy()
X_melted['class'] = y.map(dict(enumerate(target_names)))
X_melted_long = X_melted.melt(id_vars='class', var_name='Feature', value_name='Value')
short_names = [f.split(' ')[0] for f in feature_names]
X_melted_long['Feature'] = X_melted_long['Feature'].replace(dict(zip(feature_names, short_names)))
sns.boxplot(data=X_melted_long, x='Feature', y='Value', hue='class', ax=ax)
ax.set_title('Box Plots by Class', fontsize=12, fontweight='bold')
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

# 4. Pairplot (in a separate figure)
plt.tight_layout()
plt.savefig('eda_iris.png', dpi=150, bbox_inches='tight')
plt.close()

# Pairplot
df_pair = X.copy()
df_pair.columns = [f.split(' ')[0] for f in feature_names]
df_pair['class'] = y.map(dict(enumerate(target_names)))
pair_fig = sns.pairplot(df_pair, hue='class', diag_kind='kde', plot_kws={'alpha': 0.7})
pair_fig.fig.suptitle('Pairplot of Iris Features', y=1.02, fontsize=14, fontweight='bold')
plt.savefig('pairplot_iris.png', dpi=150, bbox_inches='tight')
plt.close()

# =========================================================
# Baseline Model (All Features)
# =========================================================
print("\n" + "=" * 60)
print("BASELINE MODEL - ALL 4 FEATURES")
print("=" * 60)

X_np = iris.data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_np)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, iris.target, test_size=0.3, random_state=42, stratify=iris.target
)

baseline_svm = SVC(kernel='rbf', C=1.0, random_state=42)
baseline_svm.fit(X_train, y_train)
y_pred_baseline = baseline_svm.predict(X_test)
accuracy_baseline = accuracy_score(y_test, y_pred_baseline)

cv_scores_baseline = cross_val_score(baseline_svm, X_scaled, iris.target, cv=5)

print(f"\nBaseline SVM (All Features):")
print(f"  Test Accuracy:  {accuracy_baseline:.4f}")
print(f"  CV Accuracy:    {cv_scores_baseline.mean():.4f} ± {cv_scores_baseline.std():.4f}")

# =========================================================
# c-i. Univariate Feature Selection (SelectKBest)
# =========================================================
print("\n" + "=" * 60)
print("FEATURE SELECTION - UNIVARIATE (SelectKBest with f_classif)")
print("=" * 60)

selector_uni = SelectKBest(score_func=f_classif, k=2)
selector_uni.fit(X_np, iris.target)

uni_scores = selector_uni.scores_
uni_pvalues = selector_uni.pvalues_
selected_uni_mask = selector_uni.get_support()

print("\nFeature Scores (F-statistic):")
for name, score, pval, selected in zip(feature_names, uni_scores, uni_pvalues, selected_uni_mask):
    print(f"  {name:<30} Score: {score:8.2f}   p-value: {pval:.4f}   {'✓ Selected' if selected else '✗ Rejected'}")

selected_uni_features = [feature_names[i] for i in range(4) if selected_uni_mask[i]]
print(f"\nSelected Features (k=2): {selected_uni_features}")

X_uni = selector_uni.transform(X_np)
X_uni_scaled = scaler.fit_transform(X_uni)
X_train_uni, X_test_uni, y_train_uni, y_test_uni = train_test_split(
    X_uni_scaled, iris.target, test_size=0.3, random_state=42, stratify=iris.target
)

svm_uni = SVC(kernel='rbf', C=1.0, random_state=42)
svm_uni.fit(X_train_uni, y_train_uni)
y_pred_uni = svm_uni.predict(X_test_uni)
accuracy_uni = accuracy_score(y_test_uni, y_pred_uni)
cv_scores_uni = cross_val_score(svm_uni, X_uni_scaled, iris.target, cv=5)

print(f"\nSVM with Univariate Selected Features:")
print(f"  Test Accuracy:  {accuracy_uni:.4f}")
print(f"  CV Accuracy:    {cv_scores_uni.mean():.4f} ± {cv_scores_uni.std():.4f}")

# =========================================================
# c-ii. Feature Importance using Random Forest
# =========================================================
print("\n" + "=" * 60)
print("FEATURE SELECTION - RANDOM FOREST IMPORTANCE")
print("=" * 60)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_np, iris.target)

rf_importances = rf.feature_importances_
rf_sorted_idx = np.argsort(rf_importances)[::-1]

print("\nFeature Importances (Random Forest):")
for rank, idx in enumerate(rf_sorted_idx):
    print(f"  {rank+1}. {feature_names[idx]:<30} Importance: {rf_importances[idx]:.4f}")

# Select top 2 features by RF importance
top2_rf_idx = rf_sorted_idx[:2]
selected_rf_features = [feature_names[i] for i in top2_rf_idx]
print(f"\nSelected Top-2 Features: {selected_rf_features}")

X_rf = X_np[:, top2_rf_idx]
X_rf_scaled = scaler.fit_transform(X_rf)
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(
    X_rf_scaled, iris.target, test_size=0.3, random_state=42, stratify=iris.target
)

svm_rf = SVC(kernel='rbf', C=1.0, random_state=42)
svm_rf.fit(X_train_rf, y_train_rf)
y_pred_rf = svm_rf.predict(X_test_rf)
accuracy_rf = accuracy_score(y_test_rf, y_pred_rf)
cv_scores_rf = cross_val_score(svm_rf, X_rf_scaled, iris.target, cv=5)

print(f"\nSVM with RF-Selected Features:")
print(f"  Test Accuracy:  {accuracy_rf:.4f}")
print(f"  CV Accuracy:    {cv_scores_rf.mean():.4f} ± {cv_scores_rf.std():.4f}")

# =========================================================
# c-iii. Recursive Feature Elimination (RFE) using SVM
# =========================================================
print("\n" + "=" * 60)
print("FEATURE SELECTION - RFE WITH SVM")
print("=" * 60)

# Use a linear SVM for RFE (requires coef_ attribute)
svm_linear = SVC(kernel='linear', C=1.0, random_state=42)
rfe = RFE(estimator=svm_linear, n_features_to_select=2, step=1)
rfe.fit(X_scaled, iris.target)

rfe_support = rfe.support_
rfe_ranking = rfe.ranking_

print("\nRFE Feature Rankings:")
for name, supported, rank in zip(feature_names, rfe_support, rfe_ranking):
    print(f"  {name:<30} Rank: {rank}   {'✓ Selected' if supported else '✗ Eliminated'}")

selected_rfe_features = [feature_names[i] for i in range(4) if rfe_support[i]]
print(f"\nSelected Features (n=2): {selected_rfe_features}")

X_rfe = rfe.transform(X_scaled)
X_train_rfe, X_test_rfe, y_train_rfe, y_test_rfe = train_test_split(
    X_rfe, iris.target, test_size=0.3, random_state=42, stratify=iris.target
)

svm_rfe = SVC(kernel='rbf', C=1.0, random_state=42)
svm_rfe.fit(X_train_rfe, y_train_rfe)
y_pred_rfe = svm_rfe.predict(X_test_rfe)
accuracy_rfe = accuracy_score(y_test_rfe, y_pred_rfe)
cv_scores_rfe = cross_val_score(svm_rfe, X_rfe, iris.target, cv=5)

print(f"\nSVM with RFE-Selected Features:")
print(f"  Test Accuracy:  {accuracy_rfe:.4f}")
print(f"  CV Accuracy:    {cv_scores_rfe.mean():.4f} ± {cv_scores_rfe.std():.4f}")

# =========================================================
# e. Compare Model Performance Before and After Feature Selection
# =========================================================
print("\n" + "=" * 60)
print("PERFORMANCE COMPARISON - BEFORE vs AFTER FEATURE SELECTION")
print("=" * 60)

methods = ['Baseline\n(All 4)', 'Univariate\n(Top 2)', 'RF Importance\n(Top 2)', 'RFE w/ SVM\n(Top 2)']
test_accs = [accuracy_baseline, accuracy_uni, accuracy_rf, accuracy_rfe]
cv_means = [cv_scores_baseline.mean(), cv_scores_uni.mean(), cv_scores_rf.mean(), cv_scores_rfe.mean()]
cv_stds = [cv_scores_baseline.std(), cv_scores_uni.std(), cv_scores_rf.std(), cv_scores_rfe.std()]

print(f"\n{'Method':<25} {'Test Acc':>10} {'CV Mean':>10} {'CV Std':>10} {'Features Used'}")
print("-" * 85)
comparisons = [
    ('Baseline (All 4)', accuracy_baseline, cv_scores_baseline.mean(), cv_scores_baseline.std(), ', '.join(feature_names)),
    ('Univariate (Top 2)', accuracy_uni, cv_scores_uni.mean(), cv_scores_uni.std(), ', '.join(selected_uni_features)),
    ('RF Importance (Top 2)', accuracy_rf, cv_scores_rf.mean(), cv_scores_rf.std(), ', '.join(selected_rf_features)),
    ('RFE w/ SVM (Top 2)', accuracy_rfe, cv_scores_rfe.mean(), cv_scores_rfe.std(), ', '.join(selected_rfe_features)),
]

for name, ta, cv_m, cv_s, feats in comparisons:
    print(f"{name:<25} {ta:>10.4f} {cv_m:>10.4f} {cv_s:>10.4f}   {feats}")

# =========================================================
# Visualizations for Feature Selection
# =========================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot 1: Univariate F-scores
colors = ['steelblue' if not s else 'tomato' for s in selected_uni_mask]
short_labels = [f.split(' ')[0] for f in feature_names]
bars = axes[0].bar(short_labels, uni_scores, color=colors, edgecolor='black')
axes[0].set_title('Univariate F-scores\n(red = selected)', fontsize=13, fontweight='bold')
axes[0].set_ylabel('F-score')
axes[0].set_xlabel('Feature')
axes[0].grid(True, alpha=0.3, axis='y')
for bar, score in zip(bars, uni_scores):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{score:.1f}', ha='center', fontsize=10)

# Plot 2: RF Feature Importance
rf_colors = ['tomato' if i in top2_rf_idx else 'steelblue' for i in range(4)]
bars2 = axes[1].bar(short_labels, rf_importances, color=rf_colors, edgecolor='black')
axes[1].set_title('Random Forest Importance\n(red = selected top-2)', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Importance Score')
axes[1].set_xlabel('Feature')
axes[1].grid(True, alpha=0.3, axis='y')
for bar, imp in zip(bars2, rf_importances):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f'{imp:.3f}', ha='center', fontsize=10)

# Plot 3: Model Accuracy Comparison
x = np.arange(len(methods))
width = 0.35
bars3 = axes[2].bar(x - width/2, test_accs, width, label='Test Accuracy', color='steelblue', edgecolor='black')
bars4 = axes[2].bar(x + width/2, cv_means, width, label='CV Mean Accuracy', color='darkorange',
                    edgecolor='black', yerr=cv_stds, capsize=4)
axes[2].set_title('Model Accuracy Comparison\nBefore vs After Feature Selection', fontsize=13, fontweight='bold')
axes[2].set_ylabel('Accuracy')
axes[2].set_xticks(x)
axes[2].set_xticklabels(methods, fontsize=9)
axes[2].set_ylim([0.8, 1.05])
axes[2].legend()
axes[2].grid(True, alpha=0.3, axis='y')
for bar, acc in zip(bars3, test_accs):
    axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f'{acc:.3f}', ha='center', fontsize=9)

plt.suptitle('Iris Dataset - Feature Selection Analysis', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('feature_selection_iris.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE!")
print("  Saved: eda_iris.png")
print("  Saved: pairplot_iris.png")
print("  Saved: feature_selection_iris.png")
print("=" * 60)
