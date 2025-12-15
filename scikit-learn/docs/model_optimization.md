# 🎯 Model Optimization Strategy - Based on Your Results

## 📊 Your Current Results Analysis

### Kaggle Data (Real Data):
```
Model                   Accuracy  Precision  Recall    F1-Score  ROC-AUC
Logistic Regression     0.807     0.658      0.567     0.609     0.842
K-Nearest Neighbors     0.744     0.518      0.508     0.513     0.736
Decision Tree           0.794     0.630      0.545     0.585     0.828
Random Forest           0.792     0.678      0.412     0.512     0.840
```

### Synthetic Data:
```
Model                   Accuracy  Precision  Recall    F1-Score  ROC-AUC
Logistic Regression     0.793     0.632      0.426     0.509     0.765
K-Nearest Neighbors     0.918     0.878      0.782     0.827     0.940
Decision Tree           0.800     0.636      0.485     0.551     0.808
Random Forest           0.823     0.826      0.376     0.517     0.878
```

---

## 🔍 Key Observations

### Critical Finding #1: Logistic Regression Wins on RECALL! ⭐
```
Logistic Regression has the HIGHEST recall (0.567) for real data!
This means it catches 56.7% of churners - best among all models.
```

**Why this matters for churn**:
- Recall = catching actual churners
- Missing a churner (false negative) is costly
- Better to have false alarms than miss churners

### Finding #2: Real Data is Harder (Expected!)
```
Synthetic: KNN gets 78% recall (too easy!)
Real Data: Best recall is only 57% (realistic!)
```

### Finding #3: Random Forest Underperforms on Recall
```
Random Forest: Only 41% recall on real data
This is BAD for churn prediction!
Likely needs hyperparameter tuning.
```

### Finding #4: Trade-offs Exist
```
Logistic Regression: Best recall (57%) but lower precision (66%)
Random Forest: Best precision (68%) but worst recall (41%)
```

---

## 🎯 OPTIMIZATION STRATEGY

### Priority 1: Improve Logistic Regression (Already Best!) ⭐

Since LogReg has best recall, let's make it even better!

#### Step 1.1: Adjust Classification Threshold
```python
# Default threshold = 0.5
# For churn, we want to catch MORE positives
# Try lower thresholds to increase recall

from sklearn.metrics import precision_recall_curve
import numpy as np

# Get probabilities
y_proba = model.predict_proba(X_test)[:, 1]

# Calculate precision and recall for different thresholds
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

# Find threshold that gives us target recall (e.g., 70%)
target_recall = 0.70
idx = np.argmin(np.abs(recalls - target_recall))
optimal_threshold = thresholds[idx]

print(f"Optimal threshold for {target_recall:.0%} recall: {optimal_threshold:.3f}")

# Predict with new threshold
y_pred_optimized = (y_proba >= optimal_threshold).astype(int)

# Evaluate
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_optimized))
```

**Expected Result**: Recall increases from 57% → 70%+ (with some precision trade-off)

---

#### Step 1.2: Hyperparameter Tuning for Logistic Regression
```python
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

# Parameter grid
param_grid = {
    'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization
    'classifier__penalty': ['l1', 'l2'],  # Regularization type
    'classifier__solver': ['liblinear', 'saga'],  # Solvers that support l1
    'classifier__class_weight': ['balanced', {0: 1, 1: 2}, {0: 1, 1: 3}]  # Handle imbalance
}

# Grid search optimizing for RECALL
grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='recall',  # ← Optimize for recall!
    n_jobs=-1,
    verbose=1
)

# Fit
grid.fit(X_train, y_train)

# Best parameters
print(f"Best parameters: {grid.best_params_}")
print(f"Best recall: {grid.best_score_:.3f}")

# Evaluate on test set
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))
```

**Expected Result**: Recall improves from 57% → 65%+

---

### Priority 2: Fix Random Forest (Worst Recall!) 🔧

Random Forest has terrible recall (41%) - this needs fixing!

#### Step 2.1: Tune Random Forest Hyperparameters
```python
from sklearn.ensemble import RandomForestClassifier

# Parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'class_weight': ['balanced', 'balanced_subsample', {0: 1, 1: 2}, {0: 1, 1: 3}]
}

# Random search (faster than grid search)
from sklearn.model_selection import RandomizedSearchCV

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    n_iter=50,  # Try 50 combinations
    cv=5,
    scoring='recall',  # Optimize for recall
    n_jobs=-1,
    verbose=1,
    random_state=42
)

# Fit
random_search.fit(X_train, y_train)

# Best model
print(f"Best parameters: {random_search.best_params_}")
print(f"Best recall: {random_search.best_score_:.3f}")

# Evaluate
best_rf = random_search.best_estimator_
y_pred = best_rf.predict(X_test)
print(classification_report(y_test, y_pred))
```

**Expected Result**: Recall improves from 41% → 60%+

---

### Priority 3: Cross-Validation for Robust Evaluation 📊

Your intuition is correct! Cross-validation is essential.

#### Step 3.1: Implement K-Fold Cross-Validation
```python
from sklearn.model_selection import cross_val_score, cross_validate
import numpy as np

# Define models
models = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(C=1.0, class_weight='balanced', 
                                         max_iter=1000, random_state=42))
    ]),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=15,
                                           class_weight='balanced', 
                                           random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=10, 
                                           class_weight='balanced',
                                           random_state=42)
}

# Cross-validation with multiple metrics
scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

results = {}
for name, model in models.items():
    print(f"\n{'='*70}")
    print(f"Cross-Validating: {name}")
    print('='*70)
    
    # Perform cross-validation
    cv_results = cross_validate(
        model, X_train, y_train,
        cv=5,  # 5-fold cross-validation
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1
    )
    
    # Store results
    results[name] = cv_results
    
    # Print results
    for metric in scoring:
        test_scores = cv_results[f'test_{metric}']
        train_scores = cv_results[f'train_{metric}']
        
        print(f"{metric:12s}: "
              f"Test: {test_scores.mean():.3f} (±{test_scores.std():.3f})  "
              f"Train: {train_scores.mean():.3f} (±{train_scores.std():.3f})")
    
    # Check for overfitting
    recall_gap = cv_results['train_recall'].mean() - cv_results['test_recall'].mean()
    if recall_gap > 0.1:
        print(f"⚠️  WARNING: Possible overfitting (recall gap: {recall_gap:.3f})")
```

**Why cross-validation matters**:
- Single train-test split can be lucky/unlucky
- CV gives you confidence intervals (mean ± std)
- Detects overfitting
- More robust model selection

---

### Priority 4: Handle Class Imbalance Better 🎯

Your churn data is imbalanced (26.5% churn). Let's handle this better!

#### Step 4.1: SMOTE (Synthetic Minority Over-sampling)
```python
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Create pipeline with SMOTE
pipeline = ImbPipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

# Train
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))
```

**Expected Result**: Recall increases significantly (60%+ → 70%+)

---

#### Step 4.2: Adjust Class Weights More Aggressively
```python
# Try different weight ratios
class_weights = [
    'balanced',
    {0: 1, 1: 2},  # 2x weight for minority
    {0: 1, 1: 3},  # 3x weight for minority
    {0: 1, 1: 4},  # 4x weight for minority
]

for weights in class_weights:
    model = LogisticRegression(class_weight=weights, max_iter=1000, random_state=42)
    
    # Use pipeline with scaling
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', model)
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    
    print(f"Weights: {weights}")
    print(f"  Recall: {recall:.3f}, Precision: {precision:.3f}")
    print()
```

**Expected Result**: Find optimal weight that maximizes recall

---

### Priority 5: Ensemble Methods (Advanced) 🚀

Combine multiple models for better performance!

#### Step 5.1: Voting Classifier
```python
from sklearn.ensemble import VotingClassifier

# Create individual models
lr = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(C=1.0, class_weight={0:1, 1:3}, 
                                     max_iter=1000, random_state=42))
])

rf = RandomForestClassifier(n_estimators=200, max_depth=15, 
                            class_weight='balanced', random_state=42)

dt = DecisionTreeClassifier(max_depth=10, class_weight='balanced', 
                            random_state=42)

# Create voting classifier
voting_clf = VotingClassifier(
    estimators=[('lr', lr), ('rf', rf), ('dt', dt)],
    voting='soft',  # Use probabilities
    weights=[2, 1, 1]  # Give more weight to LogReg (best recall)
)

# Train
voting_clf.fit(X_train, y_train)

# Evaluate
y_pred = voting_clf.predict(X_test)
print(classification_report(y_test, y_pred))
```

**Expected Result**: Combines strengths of all models

---

#### Step 5.2: Stacking Classifier
```python
from sklearn.ensemble import StackingClassifier

# Base models
base_models = [
    ('lr', Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000))
    ])),
    ('rf', RandomForestClassifier(n_estimators=100, class_weight='balanced')),
    ('dt', DecisionTreeClassifier(max_depth=10, class_weight='balanced'))
]

# Meta model
meta_model = LogisticRegression(class_weight='balanced', max_iter=1000)

# Create stacking classifier
stacking_clf = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5
)

# Train
stacking_clf.fit(X_train, y_train)

# Evaluate
y_pred = stacking_clf.predict(X_test)
print(classification_report(y_test, y_pred))
```

**Expected Result**: Often best performance, recall 65%+

---

## 🎯 COMPLETE OPTIMIZATION SCRIPT

Here's a complete script that implements all optimizations:

```python
"""
Complete Model Optimization for Churn Prediction
Focuses on maximizing RECALL
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, recall_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Load your data
df = pd.read_csv('your_data.csv')
X = df.drop('churn', axis=1)
y = df['churn']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("="*70)
print("OPTIMIZATION 1: Threshold Tuning for Logistic Regression")
print("="*70)

# Train baseline LogReg
lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000))
])
lr_pipeline.fit(X_train, y_train)

# Get probabilities
y_proba = lr_pipeline.predict_proba(X_test)[:, 1]

# Try different thresholds
thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
for threshold in thresholds:
    y_pred = (y_proba >= threshold).astype(int)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    print(f"Threshold: {threshold:.2f}  →  Recall: {recall:.3f}, Precision: {precision:.3f}")

print("\n" + "="*70)
print("OPTIMIZATION 2: Hyperparameter Tuning with GridSearchCV")
print("="*70)

# Parameter grid for LogReg
param_grid = {
    'classifier__C': [0.01, 0.1, 1, 10],
    'classifier__class_weight': ['balanced', {0: 1, 1: 2}, {0: 1, 1: 3}]
}

grid = GridSearchCV(lr_pipeline, param_grid, cv=5, scoring='recall', n_jobs=-1)
grid.fit(X_train, y_train)

print(f"Best params: {grid.best_params_}")
print(f"Best CV recall: {grid.best_score_:.3f}")

y_pred = grid.best_estimator_.predict(X_test)
print("\nTest set performance:")
print(classification_report(y_test, y_pred))

print("\n" + "="*70)
print("OPTIMIZATION 3: SMOTE for Class Imbalance")
print("="*70)

smote_pipeline = ImbPipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000))
])

smote_pipeline.fit(X_train, y_train)
y_pred = smote_pipeline.predict(X_test)

print(classification_report(y_test, y_pred))

print("\n" + "="*70)
print("OPTIMIZATION 4: Cross-Validation Comparison")
print("="*70)

models = {
    'LogReg (Tuned)': grid.best_estimator_,
    'LogReg + SMOTE': smote_pipeline,
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=15, 
                                           class_weight='balanced', random_state=42)
}

for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='recall')
    print(f"{name:20s}: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

print("\n" + "="*70)
print("OPTIMIZATION 5: Voting Ensemble")
print("="*70)

voting_clf = VotingClassifier(
    estimators=[
        ('lr', grid.best_estimator_),
        ('rf', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))
    ],
    voting='soft',
    weights=[2, 1]  # More weight to LogReg
)

voting_clf.fit(X_train, y_train)
y_pred = voting_clf.predict(X_test)

print(classification_report(y_test, y_pred))

print("\n" + "="*70)
print("FINAL RECOMMENDATION")
print("="*70)

# Compare all approaches
results = {
    'Baseline LogReg': lr_pipeline,
    'Tuned LogReg': grid.best_estimator_,
    'LogReg + SMOTE': smote_pipeline,
    'Voting Ensemble': voting_clf
}

best_recall = 0
best_model_name = None

for name, model in results.items():
    y_pred = model.predict(X_test)
    recall = recall_score(y_test, y_pred)
    print(f"{name:20s}: Recall = {recall:.3f}")
    
    if recall > best_recall:
        best_recall = recall
        best_model_name = name

print(f"\n🏆 BEST MODEL: {best_model_name} with {best_recall:.3f} recall")
```

---

## 📊 Expected Results After Optimization

### Before Optimization:
```
Logistic Regression: Recall = 0.567 (56.7%)
```

### After Optimization:
```
1. Threshold tuning:       Recall = 0.650-0.700 (65-70%)
2. Hyperparameter tuning:  Recall = 0.600-0.650 (60-65%)
3. SMOTE:                  Recall = 0.680-0.750 (68-75%)
4. Voting Ensemble:        Recall = 0.650-0.700 (65-70%)
5. Stacking:               Recall = 0.670-0.720 (67-72%)
```

**Target**: Get recall to 70%+ (catching 70% of churners)

---

## 🎯 Recommended Optimization Order

### Week 1: Quick Wins
1. ✅ Threshold tuning (30 min) → Easy 10% recall boost
2. ✅ Adjust class weights (30 min) → Try {0:1, 1:3}
3. ✅ Implement cross-validation (1 hour) → Robust evaluation

### Week 2: Hyperparameter Tuning
1. ✅ GridSearchCV for LogReg (2 hours) → Optimized parameters
2. ✅ RandomizedSearchCV for RF (2 hours) → Fix poor recall
3. ✅ Document best parameters (1 hour)

### Week 3: Advanced Techniques
1. ✅ SMOTE implementation (1 hour) → Better handling of imbalance
2. ✅ Voting ensemble (1 hour) → Combine models
3. ✅ Final comparison (1 hour) → Choose best approach

---

## 💡 Pro Tips

### Tip 1: Business Context Matters
```python
# For churn prediction, ask:
# "What's the cost of missing a churner vs false alarm?"

# If missing churner costs $1000, false alarm costs $50:
# → Optimize for HIGH recall, accept lower precision

# Rule of thumb for churn:
# Target: Recall > 70%, Precision > 60%
```

### Tip 2: Threshold is Powerful
```python
# Easiest way to boost recall:
# Just lower the classification threshold!

# threshold = 0.5  → Recall = 57%
# threshold = 0.4  → Recall = 65%
# threshold = 0.3  → Recall = 75%

# Find optimal threshold based on business needs
```

### Tip 3: Cross-Validation is Non-Negotiable
```python
# ALWAYS use cross-validation for:
# - Model selection
# - Hyperparameter tuning
# - Final evaluation

# Single train-test split can be misleading!
```

### Tip 4: Monitor Train vs Test
```python
# Watch for overfitting:
train_recall = 0.85
test_recall = 0.57
# → Gap of 0.28 indicates overfitting!

# Solutions:
# - Regularization (lower C for LogReg)
# - Reduce model complexity (lower max_depth for trees)
# - More data
# - Cross-validation
```

---

## 🎓 What You're Learning

### Skills You're Building:
1. ✅ **Model evaluation** - You correctly identified recall as key metric
2. ✅ **Model comparison** - You compared 4 different models
3. ✅ **Critical thinking** - You asked "how to optimize" not just "what's the answer"
4. 🚀 **Hyperparameter tuning** - Next skill to master
5. 🚀 **Cross-validation** - Making evaluation robust
6. 🚀 **Handling imbalance** - SMOTE, class weights, threshold tuning

**This is the difference between completing a course and becoming an ML engineer!** 🎯

---

## ✅ Your Next Actions

### Tomorrow:
1. [ ] Implement threshold tuning (start with 0.3, 0.35, 0.4)
2. [ ] Try class_weight={0:1, 1:3} in LogReg
3. [ ] Document new recall scores

### This Week:
1. [ ] Implement GridSearchCV for LogReg
2. [ ] Fix Random Forest with hyperparameter tuning
3. [ ] Add cross-validation to your comparison

### Expected Outcome:
```
Current:  Recall = 0.567 (catching 57% of churners)
Target:   Recall = 0.700+ (catching 70%+ of churners)
```

---

**Remember**: You're already ahead of most people by:
1. Using real data ✅
2. Focusing on the right metric (recall) ✅
3. Asking how to optimize ✅

Now let's make these models production-ready! 🚀