# 📄 ONE-PAGE CHEAT SHEET
## Classification Metrics & Workflow

---

## 🎯 WHEN TO USE EACH METRIC

| Metric | Formula | Use When | Example |
|--------|---------|----------|---------|
| **Accuracy** | (TP+TN) / Total | Classes balanced | General classification |
| **Precision** | TP / (TP+FP) | Minimize false alarms | Spam filter (don't block good emails) |
| **Recall** | TP / (TP+FN) | Catch all positives | Cancer detection (can't miss cases) |
| **F1-Score** | 2×(P×R)/(P+R) | Balance needed | Most real problems |
| **ROC-AUC** | Area under curve | Compare models | Model selection |

**For Churn**: Prioritize **Recall** (don't miss churners) + check **F1** for balance

---

## 🔄 COMPLETE ML WORKFLOW

```python
# 1. LOAD DATA
df = pd.read_csv('data.csv')

# 2. SPLIT FEATURES & TARGET
X = df.drop('churn', axis=1)
y = df['churn']

# 3. TRAIN-TEST SPLIT (ALWAYS!)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. SCALE (for LogReg, KNN only)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # FIT on train
X_test_scaled = scaler.transform(X_test)        # TRANSFORM test

# 5. TRAIN MODEL
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 6. PREDICT
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# 7. EVALUATE
from sklearn.metrics import accuracy_score, precision_score, recall_score
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall: {recall_score(y_test, y_pred):.3f}")

# 8. CONFUSION MATRIX
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
# [[TN, FP],
#  [FN, TP]]
```

---

## 🚨 COMMON MISTAKES

| Mistake | Why It's Wrong | Correct Way |
|---------|----------------|-------------|
| ❌ Evaluate on training data | Overestimates performance | Use test set |
| ❌ Fit scaler on test data | Data leakage | Fit on train only |
| ❌ Use only accuracy | Misleading for imbalanced data | Use multiple metrics |
| ❌ No stratify in split | Imbalanced train/test | Add stratify=y |
| ❌ No random_state | Results not reproducible | Set random_state=42 |

---

## 📊 INTERPRETING CONFUSION MATRIX

```
                 Predicted
                 No    Yes
Actual No    [[  TN  |  FP  ]]
Actual Yes   [[  FN  |  TP  ]]
```

- **TN (True Negative)**: Correctly predicted NO churn ✅
- **FP (False Positive)**: Predicted churn, but didn't happen ⚠️
- **FN (False Negatives)**: Missed a churner! 🚨 (WORST for churn)
- **TP (True Positive)**: Correctly predicted churn ✅

---

## 🎓 WHICH ALGORITHM WHEN?

| Algorithm | Pros | Cons | When to Use |
|-----------|------|------|-------------|
| **Logistic Regression** | Fast, interpretable | Linear only | Need interpretability |
| **Decision Tree** | Interpretable, no scaling | Overfits easily | Baseline model |
| **Random Forest** | Robust, good default | Slow, black box | Best starting point |
| **KNN** | Simple, no training | Slow predictions | Small datasets |

**Rule of thumb**: Start with **Random Forest**, then try others.

---

## 🔧 QUICK COMMANDS

```bash
# Install dependencies
pip install -r requirements.txt

# Run reference (see all concepts)
python src/classification_metrics_reference.py

# Run complete pipeline
python src/train_model.py

# Open notebook
jupyter notebook notebooks/01_classification_exercise.ipynb

# Save model
import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
```

---

## 💡 QUICK TROUBLESHOOTING

| Problem | Solution |
|---------|----------|
| ImportError | `pip install scikit-learn pandas numpy` |
| Low accuracy | Try different model, add features, tune hyperparameters |
| High train, low test | Overfitting! Reduce complexity, add regularization |
| Low recall | Lower threshold, use class_weight='balanced' |
| Slow training | Reduce n_estimators, use smaller dataset for testing |

---

## 🎯 3-STEP PROJECT TEMPLATE

### Step 1: Prepare (5 lines)
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Step 2: Train (3 lines)
```python
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### Step 3: Evaluate (1 line)
```python
ClassificationEvaluator(y_test, y_pred, y_pred_proba).print_evaluation_report()
```

---

## 📚 FILES QUICK REFERENCE

| File | Purpose | When to Use |
|------|---------|-------------|
| `classification_metrics_reference.py` | Concept reference | Forgot how metrics work? |
| `train_model.py` | Complete pipeline | Building new classifier? |
| `01_classification_exercise.ipynb` | Practice | Learning step-by-step? |
| `README.md` | Documentation | Need detailed info? |
| `QUICK_START.md` | 1-hour guide | Just starting? |

---

## 🚀 YOUR NEXT 3 ACTIONS

1. **RUN**: `python src/classification_metrics_reference.py`
2. **READ**: Watch the output and visualizations
3. **EXPERIMENT**: Change one parameter and run again

---

## 🏆 SUCCESS CHECKLIST

- [ ] Can explain accuracy vs precision vs recall
- [ ] Know when to use each metric
- [ ] Can interpret confusion matrix
- [ ] Understand why train-test split is needed
- [ ] Can build classifier from scratch
- [ ] Can evaluate model properly
- [ ] Ready to deploy (Week 2 goal)

---

**Print this page and keep it next to your computer!** 📌

**Remember**: Machine Learning is 20% algorithms, 80% data preparation and evaluation! 🎯
