# ============================================
# IMPORT LIBRARIES
# ============================================

import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# Ensemble methods
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, StackingClassifier


# ============================================
# LOAD DATASET
# ============================================

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# ============================================
# DEFINE CLASSIFIERS
# ============================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier()
}

results = {}

# ============================================
# EVALUATE ALL CLASSIFIERS
# ============================================

print("Classifier Performance:\n")

for name, model in models.items():

    scores = cross_val_score(model, X, y, cv=5)

    results[name] = scores.mean()

    print(name, "Accuracy:", scores.mean())

# ============================================
# SELECT TOP 4 CLASSIFIERS
# ============================================

top_models = sorted(results, key=results.get, reverse=True)[:4]

print("\nTop 4 Classifiers:", top_models)

# Create classifier objects
selected_models = [(name, models[name]) for name in top_models]


# ============================================
# BAGGING
# ============================================

bagging = BaggingClassifier(estimator=models[top_models[0]], n_estimators=10)

bagging.fit(X_train,y_train)
pred = bagging.predict(X_test)

print("\nBagging Accuracy:", accuracy_score(y_test,pred))


# ============================================
# BOOSTING
# ============================================

boost = AdaBoostClassifier(estimator=models[top_models[0]], n_estimators=50)

boost.fit(X_train,y_train)
pred = boost.predict(X_test)

print("Boosting Accuracy:", accuracy_score(y_test,pred))


# ============================================
# STACKING
# ============================================

stack = StackingClassifier(
    estimators=selected_models,
    final_estimator=LogisticRegression()
)

stack.fit(X_train,y_train)
pred = stack.predict(X_test)

print("Stacking Accuracy:", accuracy_score(y_test,pred))
