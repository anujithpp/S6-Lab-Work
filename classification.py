import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# =========================================================
# a. Load Data and Preprocess
# =========================================================
print("="*60)
print("LOADING AND PREPROCESSING DATA")
print("="*60)

# Load MNIST dataset
print("Downloading/Loading MNIST dataset from OpenML... (This may take a minute)")
mnist_data = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X, y = mnist_data["data"], mnist_data["target"].astype(int)

# Split into standard train/test sets (first 60k is train, last 10k is test)
x_train_full, x_test = X[:60000], X[60000:]
y_train_full, y_test = y[:60000], y[60000:]

# Data is already flattened, just normalize to 0-1
x_train_full = x_train_full.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Split into train and validation sets
x_train, x_val, y_train, y_val = train_test_split(
    x_train_full, y_train_full, 
    test_size=0.1667, 
    random_state=42,
    stratify=y_train_full
)

print(f"Training Set: {x_train.shape}")
print(f"Validation Set: {x_val.shape}")
print(f"Test Set: {x_test.shape}")

# =========================================================
# a. Build Logistic Regression Model (Baseline)
# =========================================================
print("\n" + "="*60)
print("BUILDING BASELINE LOGISTIC REGRESSION MODEL")
print("="*60)

# Initialize and train baseline model (REMOVED: multi_class parameter)
baseline_model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    solver='lbfgs'
)

baseline_model.fit(x_train, y_train)

# Predict on validation set
y_val_pred_baseline = baseline_model.predict(x_val)

# =========================================================
# b. Evaluate Model Performance
# =========================================================
print("\n" + "="*60)
print("MODEL EVALUATION - BASELINE")
print("="*60)

# Calculate metrics (multi-class average)
accuracy_baseline = accuracy_score(y_val, y_val_pred_baseline)
precision_baseline = precision_score(y_val, y_val_pred_baseline, average='weighted')
recall_baseline = recall_score(y_val, y_val_pred_baseline, average='weighted')
f1_baseline = f1_score(y_val, y_val_pred_baseline, average='weighted')

print(f"\nBaseline Model Performance (Validation Set):")
print(f"Accuracy:  {accuracy_baseline:.4f}")
print(f"Precision: {precision_baseline:.4f}")
print(f"Recall:    {recall_baseline:.4f}")
print(f"F1 Score:  {f1_baseline:.4f}")

print("\nClassification Report:")
print(classification_report(y_val, y_val_pred_baseline))

# =========================================================
# c. Fine-tune Hyperparameters using GridSearchCV
# =========================================================
print("\n" + "="*60)
print("HYPERPARAMETER TUNING WITH GRID SEARCH CV")
print("="*60)

# Define parameter grid (REMOVED: multi_class from LogisticRegression)
param_grid = {
    'C': [0.01, 0.1, 1.0, 10.0, 100.0],  # Regularization strength
    'solver': ['lbfgs', 'saga'],          # Optimization algorithms
    'penalty': ['l2', 'l1']               # Regularization types
}

# Create grid search object
grid_search = GridSearchCV(
    LogisticRegression(max_iter=1000, random_state=42),
    param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

# Fit grid search (using subset for faster computation)
sample_indices = np.random.choice(len(x_train), 10000, replace=False)
x_train_sample = x_train[sample_indices]
y_train_sample = y_train[sample_indices]

print("\nRunning Grid Search CV (this may take a few minutes)...")
grid_search.fit(x_train_sample, y_train_sample)

print(f"\nBest Parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation Score: {grid_search.best_score_:.4f}")

# Train final model with best parameters on full training set
best_model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    **grid_search.best_params_
)

best_model.fit(x_train, y_train)

# Predict with tuned model
y_val_pred_tuned = best_model.predict(x_val)

# =========================================================
# b. Evaluate Tuned Model Performance
# =========================================================
print("\n" + "="*60)
print("MODEL EVALUATION - TUNED")
print("="*60)

accuracy_tuned = accuracy_score(y_val, y_val_pred_tuned)
precision_tuned = precision_score(y_val, y_val_pred_tuned, average='weighted')
recall_tuned = recall_score(y_val, y_val_pred_tuned, average='weighted')
f1_tuned = f1_score(y_val, y_val_pred_tuned, average='weighted')

print(f"\nTuned Model Performance (Validation Set):")
print(f"Accuracy:  {accuracy_tuned:.4f}")
print(f"Precision: {precision_tuned:.4f}")
print(f"Recall:    {recall_tuned:.4f}")
print(f"F1 Score:  {f1_tuned:.4f}")

print("\nClassification Report:")
print(classification_report(y_val, y_val_pred_tuned))

# Compare baseline vs tuned
print("\n" + "="*60)
print("PERFORMANCE COMPARISON")
print("="*60)
print(f"{'Metric':<12} {'Baseline':<12} {'Tuned':<12} {'Improvement':<12}")
print("-"*60)
print(f"{'Accuracy':<12} {accuracy_baseline:<12.4f} {accuracy_tuned:<12.4f} {accuracy_tuned - accuracy_baseline:+.4f}")
print(f"{'Precision':<12} {precision_baseline:<12.4f} {precision_tuned:<12.4f} {precision_tuned - precision_baseline:+.4f}")
print(f"{'Recall':<12} {recall_baseline:<12.4f} {recall_tuned:<12.4f} {recall_tuned - recall_baseline:+.4f}")
print(f"{'F1 Score':<12} {f1_baseline:<12.4f} {f1_tuned:<12.4f} {f1_tuned - f1_baseline:+.4f}")

# =========================================================
# Test Set Evaluation (Final)
# =========================================================
print("\n" + "="*60)
print("FINAL TEST SET EVALUATION")
print("="*60)

y_test_pred = best_model.predict(x_test)

accuracy_test = accuracy_score(y_test, y_test_pred)
precision_test = precision_score(y_test, y_test_pred, average='weighted')
recall_test = recall_score(y_test, y_test_pred, average='weighted')
f1_test = f1_score(y_test, y_test_pred, average='weighted')

print(f"\nTest Set Performance:")
print(f"Accuracy:  {accuracy_test:.4f}")
print(f"Precision: {precision_test:.4f}")
print(f"Recall:    {recall_test:.4f}")
print(f"F1 Score:  {f1_test:.4f}")

# =========================================================
# d. Visualize Decision Boundary
# =========================================================
print("\n" + "="*60)
print("VISUALIZING DECISION BOUNDARY")
print("="*60)

# Reduce dimensions to 2D using PCA for visualization
pca = PCA(n_components=2, random_state=42)
x_train_pca = pca.fit_transform(x_train)
x_val_pca = pca.transform(x_val)

print(f"\nPCA Explained Variance Ratio: {pca.explained_variance_ratio_}")
print(f"Total Variance Explained: {sum(pca.explained_variance_ratio_):.4f}")

# Train logistic regression on PCA-reduced data for visualization
model_pca = LogisticRegression(
    max_iter=1000,
    random_state=42,
    **grid_search.best_params_
)
model_pca.fit(x_train_pca, y_train)

# Create mesh grid for decision boundary
h = 0.02  # step size
x_min, x_max = x_train_pca[:, 0].min() - 1, x_train_pca[:, 0].max() + 1
y_min, y_max = x_train_pca[:, 1].min() - 1, x_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Predict on mesh grid
Z = model_pca.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Plot 1: Decision Boundary
axes[0, 0].contourf(xx, yy, Z, alpha=0.3, cmap='plt.cm.Paired')
scatter = axes[0, 0].scatter(x_val_pca[:, 0], x_val_pca[:, 1], 
                              c=y_val, cmap='plt.cm.Paired', 
                              edgecolors='k', s=20, alpha=0.7)
axes[0, 0].set_xlabel('PCA Component 1', fontsize=12)
axes[0, 0].set_ylabel('PCA Component 2', fontsize=12)
axes[0, 0].set_title('Decision Boundary (PCA 2D)', fontsize=14, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)
plt.colorbar(scatter, ax=axes[0, 0], label='Digit')

# Plot 2: Confusion Matrix
cm = confusion_matrix(y_val, y_val_pred_tuned)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1],
            xticklabels=range(10), yticklabels=range(10))
axes[0, 1].set_xlabel('Predicted Label', fontsize=12)
axes[0, 1].set_ylabel('True Label', fontsize=12)
axes[0, 1].set_title('Confusion Matrix', fontsize=14, fontweight='bold')

# Plot 3: Accuracy by Digit
accuracy_per_digit = []
for i in range(10):
    mask = y_val == i
    acc = accuracy_score(y_val[mask], y_val_pred_tuned[mask])
    accuracy_per_digit.append(acc)

axes[1, 0].bar(range(10), accuracy_per_digit, color='steelblue', edgecolor='black')
axes[1, 0].set_xlabel('Digit', fontsize=12)
axes[1, 0].set_ylabel('Accuracy', fontsize=12)
axes[1, 0].set_title('Accuracy by Digit', fontsize=14, fontweight='bold')
axes[1, 0].set_xticks(range(10))
axes[1, 0].set_ylim([0, 1.1])
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, v in enumerate(accuracy_per_digit):
    axes[1, 0].text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=10)

# Plot 4: Sample Predictions
sample_indices = np.random.choice(len(x_val), 10, replace=False)
sample_images = x_val[sample_indices].reshape(-1, 28, 28)
sample_true = y_val[sample_indices]
sample_pred = y_val_pred_tuned[sample_indices]

for i, ax in enumerate(axes[1, 1].flat):
    if i < 10:
        ax.imshow(sample_images[i], cmap='gray')
        color = 'green' if sample_true[i] == sample_pred[i] else 'red'
        ax.set_title(f'True: {sample_true[i]}\nPred: {sample_pred[i]}', 
                     color=color, fontsize=10, fontweight='bold')
        ax.axis('off')

plt.suptitle('Logistic Regression - MNIST Classification Analysis', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('logistic_regression_mnist_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# =========================================================
# Additional: Feature Importance (Coefficients)
# =========================================================
print("\n" + "="*60)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*60)

# Get coefficients for each class
coefficients = best_model.coef_

# Find most important pixels for each digit
print("\nTop 5 Most Important Pixels per Digit:")
print("-" * 60)

for digit in range(10):
    coef = coefficients[digit]
    top_indices = np.argsort(np.abs(coef))[-5:][::-1]
    print(f"\nDigit {digit}:")
    for idx in top_indices:
        row, col = idx // 28, idx % 28
        print(f"  Pixel ({row}, {col}): {coef[idx]:.4f}")

# Visualize coefficient weights for a few digits
fig, axes = plt.subplots(2, 5, figsize=(20, 4))
axes = axes.flatten()

for digit in range(10):
    coef_img = coefficients[digit].reshape(28, 28)
    im = axes[digit].imshow(coef_img, cmap='RdBu_r')
    axes[digit].set_title(f'Digit {digit} Coefficients', fontsize=12, fontweight='bold')
    axes[digit].axis('off')

plt.suptitle('Logistic Regression Coefficients (Feature Weights)', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('logistic_regression_coefficients.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("ANALYSIS COMPLETE!")
print("="*60)