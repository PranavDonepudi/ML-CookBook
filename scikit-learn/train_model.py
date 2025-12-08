"""
Churn Prediction Model - Main Training Script
==============================================
Applies concepts from DataCamp Supervised Learning Course to real churn prediction.

This script implements a complete ML pipeline:
1. Data loading and exploration
2. Train-test split
3. Model training (multiple algorithms)
4. Model evaluation
5. Model comparison
6. Model saving

Author: Pranav Donepudi
Date: December 2024
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Import our metrics reference
import sys
sys.path.append(str(Path(__file__).parent))
from classification_metrics_reference import ClassificationEvaluator


class ChurnPredictionPipeline:
    """
    Complete pipeline for churn prediction.
    
    This class encapsulates everything you learned in Chapter 1:
    - Data splitting
    - Model training
    - Prediction
    - Evaluation
    """
    
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.scaler = StandardScaler()
        
    def load_data(self, filepath=None):
        """
        Load churn dataset.
        
        Args:
            filepath: Path to CSV file (if None, creates synthetic data)
        """
        if filepath and Path(filepath).exists():
            print(f"📂 Loading data from {filepath}")
            self.df = pd.read_csv(filepath)
        else:
            print("📂 Creating synthetic churn dataset for demonstration")
            self.df = self._create_synthetic_data()
        
        print(f"  Dataset shape: {self.df.shape}")
        print(f"  Features: {self.df.columns.tolist()}")
        
        return self.df
    
    def _create_synthetic_data(self):
        """
        Create realistic synthetic churn data.
        """
        from sklearn.datasets import make_classification
        
        # Create features
        X, y = make_classification(
            n_samples=2000,
            n_features=10,
            n_informative=8,
            n_redundant=2,
            n_classes=2,
            weights=[0.75, 0.25],  # 25% churn rate
            random_state=self.random_state
        )
        
        # Create DataFrame with meaningful column names
        feature_names = [
            'account_length', 'international_plan', 'voice_mail_plan',
            'num_voice_messages', 'total_day_minutes', 'total_day_calls',
            'total_eve_minutes', 'total_night_minutes', 'total_intl_calls',
            'customer_service_calls'
        ]
        
        df = pd.DataFrame(X, columns=feature_names)
        df['churn'] = y
        
        return df
    
    def explore_data(self):
        """
        Basic data exploration.
        """
        print("\n📊 DATA EXPLORATION")
        print("=" * 60)
        
        print("\n1. Dataset Info:")
        print(f"   Total records: {len(self.df)}")
        print(f"   Total features: {len(self.df.columns) - 1}")
        
        print("\n2. Target Variable Distribution:")
        churn_counts = self.df['churn'].value_counts()
        churn_pct = self.df['churn'].value_counts(normalize=True) * 100
        print(f"   No Churn (0): {churn_counts[0]} ({churn_pct[0]:.1f}%)")
        print(f"   Churn (1):    {churn_counts[1]} ({churn_pct[1]:.1f}%)")
        
        print("\n3. Feature Statistics:")
        print(self.df.describe().round(2))
        
        print("\n4. Missing Values:")
        missing = self.df.isnull().sum()
        if missing.sum() == 0:
            print("   ✓ No missing values")
        else:
            print(missing[missing > 0])
    
    def prepare_data(self):
        """
        Prepare data for modeling.
        
        Key Concepts:
        - Separate features (X) from target (y)
        - Train-test split with stratification
        - Feature scaling (important for some algorithms)
        """
        print("\n🔧 PREPARING DATA")
        print("=" * 60)
        
        # Separate features and target
        X = self.df.drop('churn', axis=1)
        y = self.df['churn']
        
        print(f"\n1. Features (X): {X.shape}")
        print(f"2. Target (y): {y.shape}")
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=y  # Maintains class balance
        )
        
        print(f"\n3. Split:")
        print(f"   Training: {len(self.X_train)} samples ({len(self.X_train)/len(X)*100:.1f}%)")
        print(f"   Test: {len(self.X_test)} samples ({len(self.X_test)/len(X)*100:.1f}%)")
        
        # Feature scaling
        print(f"\n4. Scaling features...")
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        print("   ✓ StandardScaler fitted and applied")
        
        print("\n💡 Why scaling?")
        print("   • Required for: Logistic Regression, KNN, SVM")
        print("   • Not needed for: Decision Trees, Random Forest")
        print("   • Always fit on training data, transform both train & test")
    
    def train_models(self):
        """
        Train multiple models and compare.
        
        Models from DataCamp course:
        - Logistic Regression (linear classifier)
        - K-Nearest Neighbors (instance-based)
        - Decision Tree (non-linear, interpretable)
        - Random Forest (ensemble, robust)
        """
        print("\n🤖 TRAINING MODELS")
        print("=" * 60)
        
        # Define models
        models_to_train = {
            'Logistic Regression': LogisticRegression(random_state=self.random_state, max_iter=1000),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
            'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=self.random_state),
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=self.random_state)
        }
        
        results = []
        
        for name, model in models_to_train.items():
            print(f"\n📍 Training {name}...")
            
            # Choose scaled or original data
            if name in ['Logistic Regression', 'K-Nearest Neighbors']:
                X_train = self.X_train_scaled
                X_test = self.X_test_scaled
            else:
                X_train = self.X_train
                X_test = self.X_test
            
            # Train
            model.fit(X_train, self.y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Evaluate
            evaluator = ClassificationEvaluator(self.y_test, y_pred, y_pred_proba)
            metrics = evaluator.calculate_all_metrics()
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, self.y_train, cv=5)
            metrics['cv_mean'] = cv_scores.mean()
            metrics['cv_std'] = cv_scores.std()
            
            # Store
            self.models[name] = {
                'model': model,
                'metrics': metrics,
                'evaluator': evaluator
            }
            
            results.append({
                'Model': name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score'],
                'ROC-AUC': metrics['roc_auc'],
                'CV Score': f"{metrics['cv_mean']:.3f} ± {metrics['cv_std']:.3f}"
            })
            
            print(f"   ✓ Trained | Test Accuracy: {metrics['accuracy']:.3f}")
        
        # Display comparison
        print("\n" + "=" * 60)
        print("📊 MODEL COMPARISON")
        print("=" * 60)
        
        results_df = pd.DataFrame(results)
        print("\n" + results_df.to_string(index=False))
        
        # Identify best model
        best_model_name = results_df.loc[results_df['ROC-AUC'].idxmax(), 'Model']
        self.best_model = self.models[best_model_name]
        
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"   ROC-AUC Score: {self.best_model['metrics']['roc_auc']:.3f}")
        
        return results_df
    
    def evaluate_best_model(self):
        """
        Detailed evaluation of best model.
        """
        print("\n" + "=" * 60)
        print("🏆 BEST MODEL DETAILED EVALUATION")
        print("=" * 60)
        
        self.best_model['evaluator'].print_evaluation_report()
        
        print("\n📊 Generating visualizations...")
        self.best_model['evaluator'].plot_confusion_matrix()
        self.best_model['evaluator'].plot_roc_curve()
    
    def save_model(self, filepath='models/churn_model.pkl'):
        """
        Save the best model for deployment.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        model_package = {
            'model': self.best_model['model'],
            'scaler': self.scaler,
            'feature_names': self.X_train.columns.tolist(),
            'metrics': self.best_model['metrics']
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_package, f)
        
        print(f"\n💾 Model saved to: {filepath}")
        print("   Package includes: model, scaler, feature names, metrics")
    
    def predict_new_customer(self, customer_data):
        """
        Make prediction for a new customer.
        
        Args:
            customer_data: dict or DataFrame with customer features
        """
        if isinstance(customer_data, dict):
            customer_data = pd.DataFrame([customer_data])
        
        # Scale if needed
        model_name = [k for k, v in self.models.items() if v == self.best_model][0]
        if model_name in ['Logistic Regression', 'K-Nearest Neighbors']:
            customer_data_processed = self.scaler.transform(customer_data)
        else:
            customer_data_processed = customer_data
        
        # Predict
        prediction = self.best_model['model'].predict(customer_data_processed)[0]
        probability = self.best_model['model'].predict_proba(customer_data_processed)[0]
        
        print(f"\n🎯 CHURN PREDICTION")
        print(f"   Prediction: {'CHURN' if prediction == 1 else 'NO CHURN'}")
        print(f"   Churn Probability: {probability[1]:.1%}")
        print(f"   Confidence: {'High' if max(probability) > 0.8 else 'Medium' if max(probability) > 0.6 else 'Low'}")
        
        return prediction, probability


def main():
    """
    Main execution function.
    """
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║        CHURN PREDICTION MODEL - COMPLETE PIPELINE                ║
    ║     Based on DataCamp Supervised Learning with Scikit-learn      ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Initialize pipeline
    pipeline = ChurnPredictionPipeline(test_size=0.2, random_state=42)
    
    # Step 1: Load data
    df = pipeline.load_data()
    
    # Step 2: Explore data
    pipeline.explore_data()
    
    # Step 3: Prepare data
    pipeline.prepare_data()
    
    # Step 4: Train models
    results = pipeline.train_models()
    
    # Step 5: Evaluate best model
    pipeline.evaluate_best_model()
    
    # Step 6: Save model
    pipeline.save_model('../models/churn_model.pkl')
    
    # Step 7: Test prediction
    print("\n" + "=" * 60)
    print("🧪 TESTING PREDICTION ON NEW CUSTOMER")
    print("=" * 60)
    
    # Create sample customer
    sample_customer = {
        'account_length': 100,
        'international_plan': 1,
        'voice_mail_plan': 0,
        'num_voice_messages': 20,
        'total_day_minutes': 200,
        'total_day_calls': 100,
        'total_eve_minutes': 150,
        'total_night_minutes': 100,
        'total_intl_calls': 5,
        'customer_service_calls': 3
    }
    
    prediction, probability = pipeline.predict_new_customer(sample_customer)
    
    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETE!")
    print("=" * 60)
    print("\n📚 What you've implemented from DataCamp Chapter 1:")
    print("   ✓ Train-test split with stratification")
    print("   ✓ Multiple classification algorithms")
    print("   ✓ Model evaluation with multiple metrics")
    print("   ✓ Cross-validation for robust evaluation")
    print("   ✓ Model comparison and selection")
    print("   ✓ Prediction on new data")
    print("\n🎯 Next Steps:")
    print("   1. Deploy this model as an API (Flask/FastAPI)")
    print("   2. Add feature engineering")
    print("   3. Implement hyperparameter tuning")
    print("   4. Create a web interface")


if __name__ == "__main__":
    main()
