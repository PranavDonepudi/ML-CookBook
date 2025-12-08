"""
Scikit-learn Pipeline - Complete Guide
=======================================
This is the ACTUAL Pipeline from scikit-learn that you learned in DataCamp.

A Pipeline chains together multiple steps (preprocessing + model) into a single object.

Key Benefits:
1. Prevents data leakage (automatically fits on train, transforms on test)
2. Simplifies code (one fit, one predict)
3. Makes cross-validation easier
4. Enables hyperparameter tuning across all steps
5. Cleaner production code

Author: Pranav Donepudi
Date: December 2024
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline  # ← THE REAL PIPELINE!
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# PART 1: THE PROBLEM WITHOUT PIPELINE
# =============================================================================

def without_pipeline_example():
    """
    Shows the OLD WAY (without Pipeline) - more code, more mistakes possible.
    """
    print("\n" + "="*70)
    print("WITHOUT PIPELINE (The Hard Way)")
    print("="*70)
    
    # Create data
    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\n❌ WITHOUT PIPELINE - Multiple Steps:")
    print("-" * 70)
    
    # Step 1: Scale the data
    print("Step 1: Manually scale features")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Fit on train
    X_test_scaled = scaler.transform(X_test)        # Transform test
    
    # Step 2: Train model
    print("Step 2: Train model")
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Step 3: Predict
    print("Step 3: Make predictions")
    y_pred = model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.3f}")
    
    print("\n⚠️ PROBLEMS WITH THIS APPROACH:")
    print("  1. Easy to forget to scale test data")
    print("  2. Easy to accidentally fit scaler on test data (DATA LEAKAGE!)")
    print("  3. Multiple objects to manage (scaler + model)")
    print("  4. Hard to do cross-validation correctly")
    print("  5. Lots of repetitive code")
    
    return accuracy


# =============================================================================
# PART 2: THE SOLUTION WITH PIPELINE
# =============================================================================

def with_pipeline_example():
    """
    Shows the NEW WAY (with Pipeline) - cleaner, safer, better!
    """
    print("\n" + "="*70)
    print("WITH PIPELINE (The Smart Way)")
    print("="*70)
    
    # Create data
    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\n✅ WITH PIPELINE - Everything in One Object:")
    print("-" * 70)
    
    # Create pipeline - THIS IS THE KEY!
    pipeline = Pipeline([
        ('scaler', StandardScaler()),           # Step 1: Scale
        ('classifier', LogisticRegression(random_state=42))  # Step 2: Model
    ])
    
    print("Pipeline created with 2 steps:")
    print("  1. 'scaler': StandardScaler()")
    print("  2. 'classifier': LogisticRegression()")
    
    # Train - Pipeline handles everything!
    print("\nTraining pipeline (auto-scales, then trains)...")
    pipeline.fit(X_train, y_train)
    
    # Predict - Pipeline handles everything!
    print("Making predictions (auto-scales test data correctly)...")
    y_pred = pipeline.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.3f}")
    
    print("\n✅ BENEFITS:")
    print("  1. One fit() call - does everything correctly")
    print("  2. Impossible to forget to scale test data")
    print("  3. Impossible to fit scaler on test data (prevents data leakage)")
    print("  4. Single object to manage")
    print("  5. Cross-validation works seamlessly")
    print("  6. Cleaner, more maintainable code")
    
    return pipeline, accuracy


# =============================================================================
# PART 3: PIPELINE ANATOMY
# =============================================================================

def pipeline_anatomy():
    """
    Deep dive into how Pipeline works.
    """
    print("\n" + "="*70)
    print("PIPELINE ANATOMY - How It Works")
    print("="*70)
    
    # Create a pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(random_state=42))
    ])
    
    print("\n📋 Pipeline Structure:")
    print(f"  Type: {type(pipeline)}")
    print(f"  Steps: {pipeline.steps}")
    print(f"  Named steps: {list(pipeline.named_steps.keys())}")
    
    print("\n🔍 Accessing Individual Steps:")
    print(f"  Scaler: {pipeline.named_steps['scaler']}")
    print(f"  Model: {pipeline.named_steps['model']}")
    
    # Or use indexing
    print("\n🔢 Accessing by Index:")
    print(f"  First step: {pipeline[0]}")
    print(f"  Last step: {pipeline[-1]}")
    
    print("\n⚙️ What Happens When You Call fit():")
    print("  1. fit_transform() on scaler (fits and transforms training data)")
    print("  2. fit() on model (trains on transformed data)")
    
    print("\n🎯 What Happens When You Call predict():")
    print("  1. transform() on scaler (transforms test data)")
    print("  2. predict() on model (predicts on transformed data)")
    
    print("\n💡 Key Insight:")
    print("  Pipeline NEVER calls fit() on intermediate steps during predict()!")
    print("  This prevents data leakage automatically!")


# =============================================================================
# PART 4: MULTIPLE PREPROCESSING STEPS
# =============================================================================

def multi_step_pipeline():
    """
    Pipeline with multiple preprocessing steps.
    """
    print("\n" + "="*70)
    print("MULTI-STEP PIPELINE")
    print("="*70)
    
    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Pipeline with multiple steps
    pipeline = Pipeline([
        ('scaler', StandardScaler()),       # Step 1: Standardize
        ('model', RandomForestClassifier(random_state=42))  # Step 2: Model
    ])
    
    print("\n📊 Pipeline with 2 steps:")
    for i, (name, step) in enumerate(pipeline.steps, 1):
        print(f"  Step {i}: '{name}' → {step.__class__.__name__}")
    
    # Train and evaluate
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nAccuracy: {accuracy:.3f}")
    
    print("\n💡 You can add as many steps as needed:")
    print("  Pipeline([")
    print("      ('imputer', SimpleImputer()),")
    print("      ('scaler', StandardScaler()),")
    print("      ('feature_selection', SelectKBest()),")
    print("      ('model', RandomForestClassifier())")
    print("  ])")


# =============================================================================
# PART 5: PIPELINE WITH CROSS-VALIDATION
# =============================================================================

def pipeline_with_cv():
    """
    Shows how Pipeline makes cross-validation safe and easy.
    """
    print("\n" + "="*70)
    print("PIPELINE + CROSS-VALIDATION")
    print("="*70)
    
    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42))
    ])
    
    print("\n🔄 Running 5-fold cross-validation...")
    print("\nWhat happens in each fold:")
    print("  1. Split data into train and validation")
    print("  2. Fit scaler on TRAIN fold only")
    print("  3. Transform both train and validation")
    print("  4. Train model on transformed train fold")
    print("  5. Evaluate on transformed validation fold")
    
    # Cross-validation - Pipeline ensures no data leakage!
    cv_scores = cross_val_score(pipeline, X, y, cv=5)
    
    print(f"\n📈 Cross-Validation Scores: {cv_scores}")
    print(f"   Mean: {cv_scores.mean():.3f}")
    print(f"   Std:  {cv_scores.std():.3f}")
    
    print("\n✅ Pipeline Automatically Prevents Data Leakage:")
    print("  • Scaler is fit ONLY on training folds")
    print("  • Validation folds are transformed using training scaler")
    print("  • No information leaks from validation to training")


# =============================================================================
# PART 6: PIPELINE WITH HYPERPARAMETER TUNING
# =============================================================================

def pipeline_with_gridsearch():
    """
    The REAL power: tuning both preprocessing AND model hyperparameters.
    """
    print("\n" + "="*70)
    print("PIPELINE + GRID SEARCH (The Ultimate Combo!)")
    print("="*70)
    
    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42))
    ])
    
    # Define parameter grid
    # Note: Use 'stepname__parameter' syntax
    param_grid = {
        'classifier__C': [0.1, 1, 10],           # Model hyperparameter
        'classifier__penalty': ['l1', 'l2'],     # Model hyperparameter
        'classifier__solver': ['liblinear']      # Required for l1 penalty
    }
    
    print("\n🔍 Grid Search Parameters:")
    print("  Parameter naming: 'stepname__parameter'")
    print(f"  Searching: {param_grid}")
    print(f"  Total combinations: {3 * 2 * 1} = 6")
    
    # Grid search
    print("\n⚙️ Running Grid Search with 5-fold CV...")
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, verbose=0)
    grid_search.fit(X_train, y_train)
    
    # Results
    print(f"\n🏆 Best Parameters: {grid_search.best_params_}")
    print(f"   Best CV Score: {grid_search.best_score_:.3f}")
    
    # Test performance
    y_pred = grid_search.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"   Test Accuracy: {test_accuracy:.3f}")
    
    print("\n💡 What Just Happened:")
    print("  • Tried 6 different combinations")
    print("  • For each combination, did 5-fold CV")
    print("  • Properly scaled data in each fold")
    print("  • Found best hyperparameters")
    print("  • Trained final model with best params")


# =============================================================================
# PART 7: COMPARING DIFFERENT PIPELINES
# =============================================================================

def compare_pipelines():
    """
    Compare different pipeline configurations.
    """
    print("\n" + "="*70)
    print("COMPARING DIFFERENT PIPELINES")
    print("="*70)
    
    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define multiple pipelines
    pipelines = {
        'LogReg + StandardScaler': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(random_state=42))
        ]),
        
        'LogReg + MinMaxScaler': Pipeline([
            ('scaler', MinMaxScaler()),
            ('classifier', LogisticRegression(random_state=42))
        ]),
        
        'RandomForest (no scaling)': Pipeline([
            ('classifier', RandomForestClassifier(random_state=42))
        ]),
        
        'DecisionTree (no scaling)': Pipeline([
            ('classifier', DecisionTreeClassifier(max_depth=5, random_state=42))
        ])
    }
    
    print("\n📊 Testing 4 Different Pipelines:")
    print("-" * 70)
    
    results = []
    for name, pipeline in pipelines.items():
        # Train
        pipeline.fit(X_train, y_train)
        
        # Evaluate
        train_score = pipeline.score(X_train, y_train)
        test_score = pipeline.score(X_test, y_test)
        
        # Cross-validation
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
        
        results.append({
            'Pipeline': name,
            'Train Score': f"{train_score:.3f}",
            'Test Score': f"{test_score:.3f}",
            'CV Score': f"{cv_scores.mean():.3f} ± {cv_scores.std():.3f}"
        })
        
        print(f"\n{name}:")
        print(f"  Train: {train_score:.3f}, Test: {test_score:.3f}, CV: {cv_scores.mean():.3f}")
    
    # Summary table
    print("\n" + "="*70)
    print("COMPARISON TABLE")
    print("="*70)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))


# =============================================================================
# PART 8: REAL-WORLD EXAMPLE - CHURN PREDICTION WITH PIPELINE
# =============================================================================

def churn_prediction_with_pipeline():
    """
    Complete churn prediction using Pipeline - the RIGHT way!
    """
    print("\n" + "="*70)
    print("CHURN PREDICTION - PROPER PIPELINE IMPLEMENTATION")
    print("="*70)
    
    # Create synthetic churn data
    X, y = make_classification(
        n_samples=2000,
        n_features=10,
        n_classes=2,
        weights=[0.75, 0.25],
        random_state=42
    )
    
    feature_names = [
        'account_length', 'international_plan', 'voice_mail_plan',
        'num_voice_messages', 'total_day_minutes', 'total_day_calls',
        'total_eve_minutes', 'total_night_minutes', 'total_intl_calls',
        'customer_service_calls'
    ]
    
    X = pd.DataFrame(X, columns=feature_names)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"\n📊 Dataset:")
    print(f"  Training: {len(X_train)} customers")
    print(f"  Test: {len(X_test)} customers")
    print(f"  Churn rate: {y.mean()*100:.1f}%")
    
    # Create pipeline
    print("\n🔧 Creating Pipeline:")
    churn_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42))
    ])
    
    print("  Step 1: StandardScaler()")
    print("  Step 2: RandomForestClassifier()")
    
    # Train
    print("\n⚙️ Training...")
    churn_pipeline.fit(X_train, y_train)
    
    # Evaluate
    print("\n📈 Evaluation:")
    train_score = churn_pipeline.score(X_train, y_train)
    test_score = churn_pipeline.score(X_test, y_test)
    
    print(f"  Training Accuracy: {train_score:.3f}")
    print(f"  Test Accuracy: {test_score:.3f}")
    
    # Predictions
    y_pred = churn_pipeline.predict(X_test)
    y_pred_proba = churn_pipeline.predict_proba(X_test)
    
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))
    
    # Predict new customer
    print("\n🎯 Predicting New Customer:")
    new_customer = pd.DataFrame({
        'account_length': [100],
        'international_plan': [1],
        'voice_mail_plan': [0],
        'num_voice_messages': [20],
        'total_day_minutes': [200],
        'total_day_calls': [100],
        'total_eve_minutes': [150],
        'total_night_minutes': [100],
        'total_intl_calls': [5],
        'customer_service_calls': [3]
    })
    
    prediction = churn_pipeline.predict(new_customer)[0]
    probability = churn_pipeline.predict_proba(new_customer)[0]
    
    print(f"  Prediction: {'CHURN' if prediction == 1 else 'NO CHURN'}")
    print(f"  Churn Probability: {probability[1]:.1%}")
    
    # Save pipeline
    import pickle
    with open('churn_pipeline.pkl', 'wb') as f:
        pickle.dump(churn_pipeline, f)
    print("\n💾 Pipeline saved to 'churn_pipeline.pkl'")
    print("   (includes both scaler AND model!)")
    
    return churn_pipeline


# =============================================================================
# PART 9: PIPELINE CHEAT SHEET
# =============================================================================

def print_pipeline_cheatsheet():
    """
    Quick reference for Pipeline usage.
    """
    print("\n" + "="*70)
    print("PIPELINE CHEAT SHEET")
    print("="*70)
    
    print("""
📝 BASIC SYNTAX:
    
    from sklearn.pipeline import Pipeline
    
    pipeline = Pipeline([
        ('step1_name', Transformer1()),
        ('step2_name', Transformer2()),
        ('model_name', Model())
    ])

✅ PIPELINE METHODS:
    
    pipeline.fit(X_train, y_train)           # Train everything
    pipeline.predict(X_test)                  # Predict (auto-transforms)
    pipeline.score(X_test, y_test)           # Evaluate
    pipeline.predict_proba(X_test)           # Get probabilities

🔍 ACCESSING STEPS:
    
    pipeline.named_steps['step1_name']       # By name
    pipeline['step1_name']                   # Shorter syntax
    pipeline[0]                              # By index
    pipeline.steps                           # All steps as list

⚙️ HYPERPARAMETER TUNING:
    
    param_grid = {
        'scaler__param': [values],           # Scaler parameter
        'model__param': [values]             # Model parameter
    }
    
    grid_search = GridSearchCV(pipeline, param_grid, cv=5)

💾 SAVING/LOADING:
    
    import pickle
    
    # Save entire pipeline
    with open('pipeline.pkl', 'wb') as f:
        pickle.dump(pipeline, f)
    
    # Load pipeline
    with open('pipeline.pkl', 'rb') as f:
        pipeline = pickle.load(f)

🎯 COMMON PATTERNS:
    
    # Simple: Scale + Model
    Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression())
    ])
    
    # Advanced: Multiple preprocessing
    Pipeline([
        ('imputer', SimpleImputer()),
        ('scaler', StandardScaler()),
        ('feature_selection', SelectKBest(k=10)),
        ('model', RandomForestClassifier())
    ])
    
    # Minimal: Just model (no preprocessing)
    Pipeline([
        ('model', RandomForestClassifier())
    ])

⚠️ COMMON MISTAKES:
    
    ❌ pipeline.fit_transform(X_test)        # NEVER fit on test!
    ✅ pipeline.transform(X_test)            # Or just predict()
    
    ❌ param_grid = {'C': [0.1, 1, 10]}     # Missing step name
    ✅ param_grid = {'model__C': [0.1, 1]}  # Correct syntax
    
    ❌ Multiple estimators in pipeline       # Last step must be estimator
    ✅ One estimator at the end              # All others are transformers
""")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║         SCIKIT-LEARN PIPELINE - COMPLETE GUIDE                   ║
    ║  The ACTUAL Pipeline from DataCamp (not a custom class!)         ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Run all demonstrations
    print("\n🎓 Running Pipeline Demonstrations...\n")
    
    # Part 1: Problem without Pipeline
    acc_without = without_pipeline_example()
    
    # Part 2: Solution with Pipeline
    pipeline, acc_with = with_pipeline_example()
    
    # Part 3: How Pipeline works
    pipeline_anatomy()
    
    # Part 4: Multi-step Pipeline
    multi_step_pipeline()
    
    # Part 5: Pipeline + CV
    pipeline_with_cv()
    
    # Part 6: Pipeline + GridSearch
    pipeline_with_gridsearch()
    
    # Part 7: Compare pipelines
    compare_pipelines()
    
    # Part 8: Real-world example
    churn_pipeline = churn_prediction_with_pipeline()
    
    # Part 9: Cheat sheet
    print_pipeline_cheatsheet()
    
    print("\n" + "="*70)
    print("✅ ALL DEMONSTRATIONS COMPLETE!")
    print("="*70)
    print("\n💡 Key Takeaways:")
    print("  1. Pipeline = Preprocessing + Model in ONE object")
    print("  2. Prevents data leakage automatically")
    print("  3. Makes code cleaner and safer")
    print("  4. Essential for production ML")
    print("  5. Works seamlessly with CV and GridSearch")
    print("\n📁 Next Steps:")
    print("  • Update train_model.py to use Pipeline")
    print("  • Practice with different preprocessing steps")
    print("  • Use Pipeline in ALL future projects!")
