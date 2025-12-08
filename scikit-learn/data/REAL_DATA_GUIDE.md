# 🎯 Using REAL Churn Data from Kaggle

## Why Real Data Matters

**Synthetic Data** (What we have now):
- ❌ No real patterns to discover
- ❌ No missing values to handle
- ❌ No feature engineering challenges
- ❌ Can't add to portfolio
- ❌ Doesn't teach real-world skills

**Real Data** (What you should use):
- ✅ Real business patterns
- ✅ Missing values (handle them!)
- ✅ Feature engineering opportunities
- ✅ Portfolio-worthy project
- ✅ Actual ML experience

---

## 🏆 Best Kaggle Churn Datasets

### 1. Telco Customer Churn (BEST FOR LEARNING)

**Why it's perfect**:
- ✅ 7,043 customers (good size)
- ✅ 21 features (manageable)
- ✅ Clean but has challenges
- ✅ Most popular churn dataset
- ✅ Lots of tutorials to compare against

**Dataset**: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

**Features**:
- Demographics: gender, SeniorCitizen, Partner, Dependents
- Services: PhoneService, InternetService, OnlineSecurity, TechSupport
- Account: Contract, PaymentMethod, MonthlyCharges, TotalCharges
- Target: Churn (Yes/No)

**Download URL**: 
```
https://www.kaggle.com/datasets/blastchar/telco-customer-churn
```

---

### 2. Bank Customer Churn (ALTERNATIVE)

**Dataset**: [Bank Customer Churn](https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction)

**Features**:
- 10,000 customers
- Credit score, geography, gender, age
- Balance, number of products, credit card status
- Great for understanding banking churn

---

### 3. E-commerce Customer Churn (MODERN)

**Dataset**: [E-commerce Churn](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction)

**Features**:
- E-commerce specific
- Customer complaints, cashback, tenure
- Modern use case

---

## 🚀 Quick Start: Download Telco Dataset

### Option 1: Kaggle CLI (Easiest)

```bash
# Install Kaggle CLI
pip install kaggle

# Set up Kaggle API (get your API key from kaggle.com/settings)
# Download to ~/.kaggle/kaggle.json

# Download dataset
kaggle datasets download -d blastchar/telco-customer-churn

# Unzip
unzip telco-customer-churn.zip -d data/
```

### Option 2: Manual Download (Simple)

1. Go to: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
2. Click "Download" button
3. Save to `churn-prediction-project/data/`
4. Unzip: You'll get `WA_Fn-UseC_-Telco-Customer-Churn.csv`

### Option 3: Direct Download (if available)

```bash
# Create data directory
mkdir -p data

# Download (if direct link works)
# Or use the Kaggle CLI method above
```

---

## 📊 Understanding the Telco Dataset

### Features Overview

```python
import pandas as pd

df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

print(df.info())
print(df.head())
```

**Expected Output**:
```
Columns: 21
- customerID: Unique identifier
- gender: Male/Female
- SeniorCitizen: 0 or 1
- Partner: Yes/No
- Dependents: Yes/No
- tenure: Number of months with company
- PhoneService: Yes/No
- MultipleLines: Yes/No/No phone service
- InternetService: DSL/Fiber optic/No
- OnlineSecurity: Yes/No/No internet service
- OnlineBackup: Yes/No/No internet service
- DeviceProtection: Yes/No/No internet service
- TechSupport: Yes/No/No internet service
- StreamingTV: Yes/No/No internet service
- StreamingMovies: Yes/No/No internet service
- Contract: Month-to-month/One year/Two year
- PaperlessBilling: Yes/No
- PaymentMethod: Electronic check/Mailed check/Bank transfer/Credit card
- MonthlyCharges: Numeric
- TotalCharges: Numeric (has some issues!)
- Churn: Yes/No (TARGET)
```

---

## 🔧 Real-World Challenges in This Dataset

### Challenge 1: Missing Values
```python
# TotalCharges has spaces instead of numbers for some customers
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
print(f"Missing values: {df['TotalCharges'].isna().sum()}")
# Output: 11 missing values
```

### Challenge 2: Categorical Encoding
```python
# Many Yes/No and categorical features need encoding
binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 
               'PaperlessBilling', 'Churn']
multi_cols = ['InternetService', 'Contract', 'PaymentMethod']
```

### Challenge 3: Class Imbalance
```python
print(df['Churn'].value_counts(normalize=True))
# No: 73.5%, Yes: 26.5% (imbalanced!)
```

---

## 🎯 Updated Code for Real Data

I'll create an updated version that works with the Telco dataset:

```python
# Key changes needed:
# 1. Load real CSV instead of synthetic data
# 2. Handle missing values
# 3. Encode categorical features
# 4. Handle class imbalance
# 5. Feature engineering
```

---

## 📝 Data Preprocessing Steps

### Step 1: Load and Clean
```python
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Fix TotalCharges (has spaces)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Drop customerID (not useful for prediction)
df = df.drop('customerID', axis=1)

# Handle missing values
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
```

### Step 2: Encode Categorical Features
```python
from sklearn.preprocessing import LabelEncoder

# Binary encoding (Yes/No → 1/0)
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

# Gender
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

# Target variable
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# One-hot encoding for multi-category features
df = pd.get_dummies(df, columns=['InternetService', 'Contract', 'PaymentMethod', 
                                  'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                                  'DeviceProtection', 'TechSupport', 'StreamingTV',
                                  'StreamingMovies'], 
                     drop_first=True)
```

### Step 3: Feature Engineering
```python
# Create new features
df['TenureGroup'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 72], 
                            labels=['0-1 year', '1-2 years', '2-4 years', '4+ years'])

# Monthly to Total ratio
df['ChargesRatio'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1)

# Services count
service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                'TechSupport', 'StreamingTV', 'StreamingMovies']
# df['ServicesCount'] = df[service_cols].sum(axis=1)  # After one-hot encoding
```

---

## 🆕 Updated Training Script for Real Data

I'll create a new version that handles the Telco dataset properly.

**New file**: `train_model_real_data.py`

---

## 📊 Expected Results with Real Data

### With Telco Dataset:
- **Baseline Accuracy**: ~73.5% (just predicting majority class)
- **Good Model**: 78-82% accuracy
- **Great Model**: 82-85% accuracy
- **Recall for Churners**: 60-75% (very important!)

### What Makes This Hard:
1. **Class Imbalance**: Only 26.5% churn
2. **Many Categorical Features**: Need proper encoding
3. **Missing Data**: Need to handle TotalCharges
4. **Feature Interactions**: e.g., Contract type + Tenure

---

## 🎓 Learning Benefits

### What You'll Learn with Real Data:
1. **Data Cleaning**: Handle missing values, fix data types
2. **Feature Engineering**: Create meaningful features
3. **Encoding**: Handle categorical variables properly
4. **Class Imbalance**: Use techniques like class weights, SMOTE
5. **Business Understanding**: Interpret results in business context
6. **Portfolio Building**: Real project you can showcase

---

## 🚀 Action Plan

### Step 1: Download Data (5 minutes)
```bash
# Go to Kaggle, download Telco Customer Churn
# Save to: churn-prediction-project/data/
```

### Step 2: Run Updated Script (coming next)
```bash
# I'll create train_model_real_data.py
python src/train_model_real_data.py
```

### Step 3: Compare Results
- Synthetic data: Easy patterns
- Real data: Harder, more realistic
- Better learning experience!

---

## 💡 Pro Tips

### Tip 1: Start Simple
```python
# First: Get it working with basic encoding
# Then: Add feature engineering
# Finally: Optimize hyperparameters
```

### Tip 2: Understand Business Context
```python
# Churn prediction is about:
# - Identifying at-risk customers BEFORE they leave
# - Recall is more important than precision
# - False negatives (missed churners) are costly!
```

### Tip 3: Feature Importance
```python
# Random Forest shows which features matter most
# For Telco data, usually:
# - Contract type (month-to-month churns more)
# - Tenure (new customers churn more)
# - Monthly charges (high charges → more churn)
```

---

## 📁 Updated Project Structure

```
churn-prediction-project/
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv  ← REAL DATA
│   └── README.md (dataset info)
├── src/
│   ├── train_model_real_data.py  ← NEW: Works with Kaggle data
│   ├── data_preprocessing.py     ← NEW: Data cleaning functions
│   └── ... (other files)
└── notebooks/
    └── 02_real_data_exploration.ipynb  ← NEW: EDA on real data
```

---

## 🎯 Next Steps

1. **Download the Telco dataset** from Kaggle
2. Let me create updated scripts that work with real data
3. You run it and see the difference!

**Should I create the updated scripts now that handle the real Kaggle data?** 

This will include:
- Data loading and cleaning
- Proper categorical encoding
- Feature engineering
- Handling class imbalance
- Full pipeline with real data

Say yes and I'll create them! 🚀
