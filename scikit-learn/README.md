# Customer Churn Prediction Project 🎯

**Applying DataCamp "Supervised Learning with Scikit-learn" concepts to a real project**

Author: Pranav Donepudi  
Course: DataCamp - Supervised Learning with Scikit-learn (Chapter 1: Classification)

---

## 📚 What This Project Demonstrates

This project implements everything you learned in **Chapter 1: Classification**:

### ✅ Concepts Covered:
1. **Train-Test Split**: Properly splitting data to prevent overfitting
2. **Classification Algorithms**: Logistic Regression, KNN, Decision Trees, Random Forest
3. **Model Evaluation**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
4. **Confusion Matrix**: Understanding True Positives, False Positives, etc.
5. **Model Complexity**: Balancing underfitting vs overfitting
6. **Cross-Validation**: Robust model evaluation
7. **Making Predictions**: Using trained models on new data

---

## 🗂️ Project Structure

```
churn-prediction-project/
├── data/                          # Dataset storage
├── notebooks/                     # Jupyter notebooks for exploration
├── src/
│   ├── classification_metrics_reference.py  # YOUR GO-TO REFERENCE
│   └── train_model.py                       # Complete training pipeline
├── models/                        # Saved models
├── api/                          # API for deployment (future)
├── tests/                        # Unit tests
├── README.md                     # This file
└── requirements.txt              # Python dependencies
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Reference File (HIGHLY RECOMMENDED)

This file contains all the concepts with working examples:

```bash
cd src
python classification_metrics_reference.py
```

**What it does:**
- Demonstrates train-test split
- Shows model complexity tradeoff
- Runs complete churn prediction example
- Generates visualizations
- Prints detailed explanations

### 3. Run the Complete Pipeline

```bash
cd src
python train_model.py
```

**What it does:**
- Loads/creates churn dataset
- Explores data
- Trains 4 different models
- Compares their performance
- Saves best model
- Makes sample prediction

---

## 📖 How to Use This Project for Learning

### Step 1: Understand the Reference File
Open `src/classification_metrics_reference.py` and read through it:
- Each function has detailed comments
- Run it to see the output
- Experiment by changing parameters

### Step 2: Run the Complete Pipeline
Execute `src/train_model.py`:
- Watch how all concepts work together
- Note the model comparison results
- Understand why one model performs better

### Step 3: Modify and Experiment

Try these experiments to deepen understanding:

#### Experiment 1: Change Train-Test Split
```python
# In train_model.py, line ~15
pipeline = ChurnPredictionPipeline(test_size=0.3, random_state=42)  # Try 0.3 instead of 0.2
```
**Question**: How does this affect model performance?

#### Experiment 2: Adjust Model Complexity
```python
# In train_model.py, around line ~150
'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=self.random_state)  # Try 10 instead of 5
```
**Question**: Does the model overfit? Check train vs test accuracy.

#### Experiment 3: Try Different Models
Add a new model to the comparison:
```python
from sklearn.svm import SVC
'SVM': SVC(kernel='rbf', probability=True, random_state=self.random_state)
```

#### Experiment 4: Feature Engineering
Add new features to the dataset and see if performance improves.

---

## 🎯 Learning Checklist

Track your understanding:

### Basic Concepts
- [ ] I understand why we need train-test split
- [ ] I can explain what each metric means (accuracy, precision, recall, F1)
- [ ] I know when to use each metric
- [ ] I understand the confusion matrix
- [ ] I can interpret model predictions

### Intermediate Concepts
- [ ] I understand overfitting vs underfitting
- [ ] I know why we scale features
- [ ] I can compare multiple models
- [ ] I understand cross-validation
- [ ] I can select the best model based on metrics

### Advanced Concepts
- [ ] I can tune model hyperparameters
- [ ] I understand the bias-variance tradeoff
- [ ] I can make predictions on new data
- [ ] I can save and load models
- [ ] I'm ready to deploy a model

---

## 📊 Expected Output

When you run `train_model.py`, you should see:

1. **Data Exploration**: Dataset statistics and class distribution
2. **Data Preparation**: Train-test split details
3. **Model Training**: Progress for each model
4. **Model Comparison**: Table comparing all models
5. **Best Model Evaluation**: Detailed metrics and visualizations
6. **Sample Prediction**: Test on a new customer

### Example Model Comparison:
```
                  Model  Accuracy  Precision  Recall  F1-Score  ROC-AUC           CV Score
    Logistic Regression     0.820      0.750   0.650     0.696    0.850  0.815 ± 0.023
  K-Nearest Neighbors       0.800      0.700   0.620     0.658    0.830  0.795 ± 0.028
         Decision Tree      0.840      0.780   0.690     0.732    0.870  0.835 ± 0.019
        Random Forest       0.860      0.810   0.720     0.763    0.890  0.855 ± 0.015
```

---

## 🔍 Key Files Explained

### 1. `classification_metrics_reference.py`
**Purpose**: Your permanent reference for classification concepts

**Contains**:
- `ClassificationEvaluator` class: Complete evaluation toolkit
- `demonstrate_train_test_split()`: Why and how to split data
- `demonstrate_model_complexity_tradeoff()`: Visual demo of overfitting
- `complete_churn_prediction_example()`: End-to-end example

**When to use**: Anytime you need to remember how a metric works or want to see a working example

### 2. `train_model.py`
**Purpose**: Production-ready training pipeline

**Contains**:
- `ChurnPredictionPipeline` class: Complete ML workflow
- Data loading and exploration
- Multiple model training
- Model comparison and selection
- Model saving for deployment

**When to use**: When building actual churn prediction models

---

## 💡 Common Questions

### Q: Why do we need a test set?
**A**: To evaluate model performance on unseen data. Training accuracy can be misleading due to overfitting.

### Q: Which metric should I use?
**A**: 
- **Accuracy**: When classes are balanced
- **Precision**: When false positives are costly (e.g., spam detection)
- **Recall**: When false negatives are costly (e.g., disease detection)
- **F1-Score**: When you need balance between precision and recall
- **ROC-AUC**: For threshold-independent evaluation

### Q: What's a good accuracy score?
**A**: Depends on the baseline. If 90% of customers don't churn, 90% accuracy is no better than predicting "no churn" for everyone. Always compare to baseline!

### Q: How do I know if my model is overfitting?
**A**: Large gap between training and test accuracy (e.g., train: 99%, test: 75%). Use cross-validation to detect this.

---

## 🎓 Next Steps After This Project

### Week 1: Enhance This Project
1. Add feature engineering (interaction terms, polynomial features)
2. Implement hyperparameter tuning (GridSearchCV, RandomizedSearchCV)
3. Try ensemble methods (VotingClassifier, StackingClassifier)

### Week 2: Deploy the Model
1. Create Flask API (see `api/` folder - to be created)
2. Build Streamlit dashboard
3. Deploy to Heroku/Render

### Week 3: New Dataset
1. Find a new classification dataset (Kaggle)
2. Apply this entire pipeline
3. Compare results

---

## 📚 Resources

### DataCamp Courses (Completed)
- ✅ Supervised Learning with Scikit-learn - Chapter 1

### Next DataCamp Courses
- [ ] Supervised Learning with Scikit-learn - Chapters 2-4
- [ ] Feature Engineering for Machine Learning
- [ ] Model Validation in Python

### Additional Resources
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Confusion Matrix Explained](https://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/)
- [ROC Curves and AUC](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc)

---

## 🤝 Contributing

This is a learning project. Feel free to:
- Add new models
- Improve visualizations
- Add more experiments
- Create better documentation

---

## 📝 Learning Log

Keep track of your progress:

**Date**: [Your Date]
- [x] Completed DataCamp Chapter 1
- [x] Ran classification_metrics_reference.py
- [ ] Ran train_model.py successfully
- [ ] Understood all metrics
- [ ] Experimented with different models
- [ ] Modified code successfully
- [ ] Ready for deployment

---

## 🏆 Success Criteria

You've mastered this project when you can:

1. ✅ Explain each metric to someone else
2. ✅ Choose appropriate metrics for different business problems
3. ✅ Build and evaluate a classification model from scratch
4. ✅ Identify and fix overfitting
5. ✅ Deploy a model (next phase)

---

## 📧 Contact

**Pranav Donepudi**
- Email: donepudipranav04@gmail.com
- LinkedIn: [Your LinkedIn]
- GitHub: [Your GitHub]

---

## 🙏 Acknowledgments

- DataCamp for the excellent Supervised Learning course
- Scikit-learn team for the amazing library
- The ML community for inspiration

---

**Remember**: The goal isn't to complete the course, it's to build projects that demonstrate your skills! 🚀
