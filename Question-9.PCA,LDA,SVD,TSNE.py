# ============================================
# IMPORT LIBRARIES
# ============================================

import numpy as np
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

# ============================================
# LOAD DATASET (WINE DATASET)
# ============================================

wine = load_wine()
X = wine.data
y = wine.target

model = SVC()

# ============================================
# CASE 1 : REDUCE TO 2 FEATURES
# ============================================

print("\n--- CASE 1 : 2 FEATURES ---")

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
score_pca = cross_val_score(model, X_pca, y, cv=5)
print("PCA Accuracy:", score_pca.mean())

# LDA
lda = LDA(n_components=2)
X_lda = lda.fit_transform(X, y)
score_lda = cross_val_score(model, X_lda, y, cv=5)
print("LDA Accuracy:", score_lda.mean())

# TSNE
tsne = TSNE(n_components=2, random_state=0)
X_tsne = tsne.fit_transform(X)
score_tsne = cross_val_score(model, X_tsne, y, cv=5)
print("TSNE Accuracy:", score_tsne.mean())

# SVD
svd = TruncatedSVD(n_components=2)
X_svd = svd.fit_transform(X)
score_svd = cross_val_score(model, X_svd, y, cv=5)
print("SVD Accuracy:", score_svd.mean())


# ============================================
# CASE 2 : REDUCE TO 3 FEATURES
# ============================================

print("\n--- CASE 2 : 3 FEATURES ---")

# PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)
score_pca = cross_val_score(model, X_pca, y, cv=5)
print("PCA Accuracy:", score_pca.mean())

# LDA (maximum components = classes-1 = 2)
lda = LDA(n_components=2)
X_lda = lda.fit_transform(X, y)
score_lda = cross_val_score(model, X_lda, y, cv=5)
print("LDA Accuracy:", score_lda.mean())

# TSNE
tsne = TSNE(n_components=3, random_state=0)
X_tsne = tsne.fit_transform(X)
score_tsne = cross_val_score(model, X_tsne, y, cv=5)
print("TSNE Accuracy:", score_tsne.mean())

# SVD
svd = TruncatedSVD(n_components=3)
X_svd = svd.fit_transform(X)
score_svd = cross_val_score(model, X_svd, y, cv=5)
print("SVD Accuracy:", score_svd.mean())
