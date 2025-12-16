# 🎯 Feature Engineering & Logistic Regression Explained

## Part 1: What is Logistic Regression? (Simple Explanation)

### The Intuition

**Think of it like scoring a test:**

Imagine you're a teacher deciding if a student will pass or fail. You look at different factors and give each a "weight":

```
Score = (5 × hours_studied) - (3 × hours_gaming) + (2 × previous_score) - (4 × absences)

If score > threshold → Student passes
If score < threshold → Student fails
```

**Logistic Regression does EXACTLY this for churn prediction!**

---

### How It Works (3 Steps)

#### Step 1: Calculate Score (Linear Combination)
```python
score = (w1 × tenure) + (w2 × monthly_charges) + (w3 × contract) + bias

# Example:
score = (0.5 × 12) + (-0.3 × 70) + (0.8 × 1) + 2.1
score = 6 - 21 + 0.8 + 2.1 = -12.1
```

#### Step 2: Convert to Probability (Sigmoid)
```python
probability = 1 / (1 + e^(-score))

# If score = -12.1:
probability = 1 / (1 + e^12.1) = 0.0005 (0.05% churn risk - very low!)

# If score = +3.5:
probability = 1 / (1 + e^-3.5) = 0.97 (97% churn risk - very high!)
```

#### Step 3: Make Decision
```python
if probability >= 0.5:
    prediction = "Churn"
else:
    prediction = "No Churn"
```

---

### Why LogReg Has Best Recall for Your Data

**Your results show LogReg wins on recall (56.7%) because:**

1. **Linear relationships work well** for churn
   - Short tenure → more churn (linear!)
   - High charges → more churn (linear!)
   - Month-to-month → more churn (linear!)

2. **Class weights help** with imbalanced data
   - `class_weight='balanced'` gives churners more importance
   - Optimizes for catching minority class (churners)

3. **Doesn't overfit** like Random Forest
   - Your RF has only 41% recall (too cautious!)
   - LogReg is simpler, generalizes better

---

## Part 2: Feature Engineering for Churn 🔧

### Why It Matters

**Without feature engineering**: Model accuracy = 80%
**With feature engineering**: Model accuracy = 85-88%

**That's 5-8% improvement - massive in production!**

---

### 7 Feature Engineering Strategies

#### 1. Aggregation Features (Count/Sum)
```python
# Total services subscribed
service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies']
df['total_services'] = df[service_cols].sum(axis=1)

# Why this helps:
# "Customers with more services are less likely to churn"
# LogReg learns: total_services has negative coefficient
```

#### 2. Ratio Features
```python
# Monthly to total charges ratio
df['monthly_total_ratio'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1)

# Services per dollar
df['services_per_dollar'] = df['total_services'] / (df['MonthlyCharges'] + 1)

# Why this helps:
# "Value perception matters - paying more but getting less = churn"
```

#### 3. Interaction Features
```python
# High-risk combination
df['high_risk'] = ((df['tenure'] < 12) & 
                  (df['MonthlyCharges'] > 70)).astype(int)

# Month-to-month + Fiber optic
df['mtm_fiber_risk'] = ((df['Contract'] == 'Month-to-month') & 
                       (df['InternetService'] == 'Fiber optic')).astype(int)

# Why this helps:
# "Combinations matter - new + expensive = very high churn"
# LogReg can't learn "AND" logic without explicit features
```

#### 4. Binning / Categorization
```python
# Tenure groups
df['tenure_group'] = pd.cut(df['tenure'], 
                           bins=[0, 12, 24, 48, 72],
                           labels=[0, 1, 2, 3])

# Charge levels
df['charge_level'] = pd.cut(df['MonthlyCharges'],
                           bins=[0, 35, 70, 120],
                           labels=['Low', 'Medium', 'High'])

# Why this helps:
# "Non-linear patterns - churn drops sharply after 12 months"
```

#### 5. Boolean Flags
```python
# High-value customer
df['is_high_value'] = (df['TotalCharges'] > df['TotalCharges'].quantile(0.75)).astype(int)

# Long-term customer  
df['is_long_term'] = (df['tenure'] > 36).astype(int)

# Low engagement
df['low_engagement'] = (df['total_services'] < 2).astype(int)

# Why this helps:
# "Clear business rules - easy for model to learn"
```

#### 6. Time-based Features
```python
# Average charges per month
df['avg_monthly_charges'] = df['TotalCharges'] / (df['tenure'] + 1)

# Customer lifetime value estimate
df['customer_ltv'] = df['tenure'] * df['MonthlyCharges']

# Why this helps:
# "Time-normalized metrics are more comparable"
```

#### 7. Domain-Specific Risk Scores
```python
# Custom churn risk score
df['risk_score'] = (
    (df['tenure'] < 12).astype(int) * 3 +           # New: +3 points
    (df['MonthlyCharges'] > 70).astype(int) * 2 +   # Expensive: +2
    (df['total_services'] < 2).astype(int) * 2 +    # Low engagement: +2
    (df['Contract'] == 'Month-to-month').astype(int) * 4  # MTM: +4
)

# Risk score ranges from 0 (lowest) to 11 (highest)

# Why this helps:
# "Encodes business knowledge directly"
```

---

### Complete Feature Engineering Function

```python
def engineer_churn_features(df):
    """
    Complete feature engineering for churn prediction.
    Returns DataFrame with new features.
    """
    df_new = df.copy()
    
    # 1. AGGREGATION
    service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies']
    df_new['total_services'] = df_new[service_cols].sum(axis=1)
    
    # 2. RATIOS
    df_new['monthly_total_ratio'] = df_new['MonthlyCharges'] / (df_new['TotalCharges'] + 1)
    df_new['services_per_dollar'] = df_new['total_services'] / (df_new['MonthlyCharges'] + 1)
    df_new['tenure_per_dollar'] = df_new['tenure'] / (df_new['MonthlyCharges'] + 1)
    
    # 3. INTERACTIONS
    df_new['tenure_x_charges'] = df_new['tenure'] * df_new['MonthlyCharges']
    df_new['senior_charges'] = df_new['SeniorCitizen'] * df_new['MonthlyCharges']
    
    # 4. BINNING
    df_new['tenure_group'] = pd.cut(df_new['tenure'],
                                   bins=[-1, 12, 24, 48, 100],
                                   labels=[0, 1, 2, 3],
                                   include_lowest=True).astype(int)
    
    df_new['charge_level'] = pd.cut(df_new['MonthlyCharges'],
                                   bins=[0, 35, 70, 150],
                                   labels=[0, 1, 2],
                                   include_lowest=True).astype(int)
    
    # 5. BOOLEAN FLAGS
    df_new['high_risk_new'] = ((df_new['tenure'] < 12) & 
                              (df_new['MonthlyCharges'] > 70)).astype(int)
    df_new['low_engagement'] = (df_new['total_services'] < 2).astype(int)
    df_new['long_term'] = (df_new['tenure'] > 36).astype(int)
    df_new['high_value'] = (df_new['TotalCharges'] > 
                           df_new['TotalCharges'].quantile(0.75)).astype(int)
    
    # 6. TIME-BASED
    df_new['avg_monthly_charges'] = df_new['TotalCharges'] / (df_new['tenure'] + 1)
    df_new['customer_ltv'] = df_new['tenure'] * df_new['MonthlyCharges']
    
    # 7. RISK SCORE
    df_new['risk_score'] = (
        (df_new['tenure'] < 12).astype(int) * 3 +
        (df_new['MonthlyCharges'] > 70).astype(int) * 2 +
        (df_new['total_services'] < 2).astype(int) * 2
    )
    
    return df_new
```

---

### Testing Feature Impact

```python
# Test each feature's impact
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Baseline (no feature engineering)
baseline_recall = cross_val_score(
    LogisticRegression(class_weight='balanced'),
    X_train, y_train, cv=5, scoring='recall'
).mean()

# With feature engineering
X_train_eng = engineer_churn_features(X_train)
improved_recall = cross_val_score(
    LogisticRegression(class_weight='balanced'),
    X_train_eng, y_train, cv=5, scoring='recall'
).mean()

print(f"Baseline: {baseline_recall:.3f}")
print(f"Improved: {improved_recall:.3f}")
print(f"Gain: +{(improved_recall - baseline_recall)*100:.1f}%")
```

---

### Feature Importance Analysis

```python
# Train model
model = LogisticRegression(class_weight='balanced')
model.fit(X_train_eng, y_train)

# Get feature importance (coefficients)
importance_df = pd.DataFrame({
    'feature': X_train_eng.columns,
    'coefficient': model.coef_[0],
    'abs_coef': np.abs(model.coef_[0])
}).sort_values('abs_coef', ascending=False)

print("\nTop 10 Features:")
print(importance_df.head(10))

# Interpretation:
# Positive coefficient → Increases churn risk
# Negative coefficient → Decreases churn risk  
# Larger |coefficient| → More important
```

---

## Part 3: Notebook → Production Code

### The Transformation

#### Notebook (Learning) ❌
```python
# Good for: Exploring, experimenting, learning
# Bad for: Reusing, deploying, maintaining

df = pd.read_csv('data.csv')
df['new_feature'] = df['a'] * df['b']
model = LogisticRegression()
model.fit(X, y)
```

#### Production (Real Work) ✅
```python
# Good for: Reusing, deploying, maintaining
# Organized into classes and methods

class ChurnPredictor:
    def load_data(self): ...
    def engineer_features(self): ...
    def train(self): ...
    def predict(self): ...
```

---

### Production Code Structure

```
production_code/
├── config/
│   └── config.yaml              # Settings, hyperparameters
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py            # Load data
│   │   └── preprocessor.py      # Clean data
│   ├── features/
│   │   ├── __init__.py
│   │   └── engineering.py       # Feature engineering
│   ├── models/
│   │   ├── __init__.py
│   │   ├── trainer.py           # Train models
│   │   └── evaluator.py         # Evaluate models
│   └── utils/
│       ├── __init__.py
│       └── logger.py            # Logging
├── tests/
│   ├── test_features.py         # Test feature engineering
│   └── test_models.py           # Test models
├── main.py                      # Entry point
├── requirements.txt
└── README.md
```

---

### Key Differences

| Aspect | Notebook | Production |
|--------|----------|------------|
| **Organization** | Cells | Classes/Modules |
| **Reusability** | Copy-paste | Import/call |
| **Testing** | Manual | Automated tests |
| **Error handling** | Crashes | Graceful failures |
| **Logging** | Print statements | Proper logging |
| **Configuration** | Hardcoded | Config files |
| **Deployment** | Can't deploy | API/service ready |

---

## Part 4: Your Next Steps

### Step 1: Add Feature Engineering (This Week)

```python
# In your current code, add:
def engineer_features(df):
    # Copy the function from above
    ...
    return df_eng

# Apply it:
df_train_eng = engineer_features(df_train)
df_test_eng = engineer_features(df_test)

# Train with engineered features:
model.fit(df_train_eng, y_train)

# Compare results:
# Before FE: Recall = 56.7%
# After FE:  Recall = 65%+ (expected)
```

### Step 2: Combine with Optimization (This Week)

```python
# Best combination:
# 1. Feature engineering
X_eng = engineer_features(X)

# 2. Tune hyperparameters
param_grid = {
    'C': [0.1, 1, 10],
    'class_weight': ['balanced', {0:1, 1:3}]
}

# 3. Optimize threshold
optimal_threshold = 0.4  # Lower than default 0.5

# Expected result: 70%+ recall!
```

### Step 3: Create Production Code (Next Week)

Once you have your best model, I'll help you transform it into:
1. Clean class-based structure
2. Proper error handling
3. Configuration management
4. Testing framework
5. API-ready code

---

## Summary

### What You Learned

1. ✅ **Logistic Regression**: How it works (scoring → sigmoid → decision)
2. ✅ **Why it works well**: Linear relationships, handles imbalance, interpretable
3. ✅ **Feature Engineering**: 7 strategies to improve models
4. ✅ **Testing features**: How to measure impact
5. ✅ **Production mindset**: Classes vs notebooks

### Your Improved Pipeline

```
Current:
Raw data → LogReg → 56.7% recall

Next (with everything):
Raw data → Feature Engineering → Optimized LogReg → 70%+ recall
                ↓
         Threshold tuning
         Class weights  
         GridSearchCV
         Cross-validation
```

### Action Items

**Tomorrow**:
- [ ] Implement feature engineering function
- [ ] Test impact on recall
- [ ] Document which features help most

**This Week**:
- [ ] Combine feature engineering + optimization
- [ ] Achieve 70%+ recall
- [ ] Document your process

**Next Week**:
- [ ] Transform to production code
- [ ] Create API
- [ ] Deploy!

---

**You're asking exactly the right questions!** Feature engineering + optimization will get you to 70%+ recall. Then we'll make it production-ready. 🚀