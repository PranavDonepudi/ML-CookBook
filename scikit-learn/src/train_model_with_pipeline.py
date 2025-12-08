"""
Churn Prediction with Scikit-learn Pipeline
============================================
UPDATED VERSION: Now uses the proper sklearn.pipeline.Pipeline!

This demonstrates the CORRECT way to build ML models using Pipeline.

Author: Pranav Donepudi  
Date: December 2024
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline  # ← THE KEY IMPORT!

# Import our metrics reference
import sys
sys.path.append(str(Path(__file__).parent))
from classification_metrics_reference import ClassificationEvaluator


class ChurnPredictionWithPipeline:
    """
    Complete churn prediction using sklearn Pipeline.
    
    Key difference from before:
    - Uses sklearn.pipeline.Pipeline (not a custom class)
    - Cleaner code
    - Safer (prevents data leakage)
    - Production-ready
    """
    
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.pipelines = {}
        self.best_pipeline = None
        
    def load_data(self, filepath=None):
        """Load churn dataset."""
        if filepath and Path(filepath).exists():
            print(f"📂 Loading data from {filepath}")
            self.df = pd.read_csv(filepath)
        else:
            print("📂 Creating synthetic churn dataset")
            self.df = self._create_synthetic_data()
        
        print(f"  Dataset shape: {self.df.shape}")
        return self.df
    
    def _create_synthetic_data(self):
        """Create synthetic churn data."""
        from sklearn.datasets import make_classification
        
        X, y = make_classification(
            n_samples=2000,
            n_features=10,
            n_informative=8,
            n_redundant=2,
            n_classes=2,
            weights=[0.75, 0.25],
            random_state=self.random_state
        )
        
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
        """Basic data exploration."""
        print("\n📊 DATA EXPLORATION")
        print("=" * 60)
        
        print(f"\n1. Dataset: {len(self.df)} records, {len(self.df.columns)-1} features")
        
        churn_counts = self.df['churn'].value_counts()
        churn_pct = self.df['churn'].value_counts(normalize=True) * 100
        print(f"\n2. Target Distribution:")
        print(f"   No Churn (0): {churn_counts[0]} ({churn_pct[0]:.1f}%)")
        print(f"   Churn (1):    {churn_counts[1]} ({churn_pct[1]:.1f}%)")
        
        print(f"\n3. Feature Statistics:")
        print(self.df.describe().round(2))
    
    def prepare_data(self):
        """Prepare data for modeling."""
        print("\n🔧 PREPARING DATA")
        print("=" * 60)
        
        X = self.df.drop('churn', axis=1)
        y = self.df['churn']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )
        
        print(f"\nTrain: {len(self.X_train)} samples ({len(self.X_train)/len(X)*100:.1f}%)")
        print(f"Test:  {len(self.X_test)} samples ({len(self.X_test)/len(X)*100:.1f}%)")
        
        print("\n💡 Using Pipeline - No manual scaling needed!")
        print("   Pipeline will handle scaling automatically")
    
    def create_pipelines(self):
        """
        Create multiple pipelines to compare.
        
        Each pipeline is a complete preprocessing + model workflow.
        """
        print("\n🔧 CREATING PIPELINES")
        print("=" * 60)
        
        # Pipeline 1: Logistic Regression with scaling
        pipe_lr = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(random_state=self.random_state, max_iter=1000))
        ])
        
        # Pipeline 2: KNN with scaling
        pipe_knn = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', KNeighborsClassifier(n_neighbors=5))
        ])
        
        # Pipeline 3: Decision Tree (no scaling needed)
        pipe_dt = Pipeline([
            ('classifier', DecisionTreeClassifier(max_depth=5, random_state=self.random_state))
        ])
        
        # Pipeline 4: Random Forest (no scaling needed)
        pipe_rf = Pipeline([
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=self.random_state
            ))
        ])
        
        self.pipelines = {
            'Logistic Regression': pipe_lr,
            'K-Nearest Neighbors': pipe_knn,
            'Decision Tree': pipe_dt,
            'Random Forest': pipe_rf
        }
        
        print("\n✅ Created 4 Pipelines:")
        for name, pipeline in self.pipelines.items():
            steps = ' → '.join([step[0] for step in pipeline.steps])
            print(f"  • {name}: {steps}")
        
        return self.pipelines
    
    def train_and_compare(self):
        """
        Train all pipelines and compare performance.
        """
        print("\n🤖 TRAINING PIPELINES")
        print("=" * 60)
        
        results = []
        
        for name, pipeline in self.pipelines.items():
            print(f"\n📍 Training {name}...")
            
            # Train - Pipeline handles everything!
            pipeline.fit(self.X_train, self.y_train)
            
            # Predict
            y_pred = pipeline.predict(self.X_test)
            y_pred_proba = pipeline.predict_proba(self.X_test)[:, 1]
            
            # Evaluate
            evaluator = ClassificationEvaluator(self.y_test, y_pred, y_pred_proba)
            metrics = evaluator.calculate_all_metrics()
            
            # Cross-validation
            cv_scores = cross_val_score(pipeline, self.X_train, self.y_train, cv=5)
            
            results.append({
                'Pipeline': name,
                'Accuracy': f"{metrics['accuracy']:.3f}",
                'Precision': f"{metrics['precision']:.3f}",
                'Recall': f"{metrics['recall']:.3f}",
                'F1-Score': f"{metrics['f1_score']:.3f}",
                'ROC-AUC': f"{metrics['roc_auc']:.3f}",
                'CV Score': f"{cv_scores.mean():.3f} ± {cv_scores.std():.3f}"
            })
            
            # Store results
            self.pipelines[name] = {
                'pipeline': pipeline,
                'metrics': metrics,
                'evaluator': evaluator
            }
            
            print(f"   ✓ Test Accuracy: {metrics['accuracy']:.3f} | CV: {cv_scores.mean():.3f}")
        
        # Display comparison
        print("\n" + "="*60)
        print("📊 PIPELINE COMPARISON")
        print("="*60)
        results_df = pd.DataFrame(results)
        print("\n" + results_df.to_string(index=False))
        
        # Select best pipeline
        best_name = max(self.pipelines.items(), 
                       key=lambda x: x[1]['metrics']['roc_auc'])[0]
        self.best_pipeline = self.pipelines[best_name]
        
        print(f"\n🏆 Best Pipeline: {best_name}")
        print(f"   ROC-AUC: {self.best_pipeline['metrics']['roc_auc']:.3f}")
        
        return results_df
    
    def tune_best_pipeline(self):
        """
        Hyperparameter tuning on the best pipeline.
        
        This shows the real power of Pipeline + GridSearch!
        """
        print("\n" + "="*60)
        print("🔍 HYPERPARAMETER TUNING")
        print("="*60)
        
        # Create a fresh Random Forest pipeline for tuning
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(random_state=self.random_state))
        ])
        
        # Define parameter grid
        # Note: Use 'stepname__parameter' syntax
        param_grid = {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [3, 5, 10],
            'classifier__min_samples_split': [2, 5]
        }
        
        print("\n⚙️ Grid Search Parameters:")
        for param, values in param_grid.items():
            print(f"   {param}: {values}")
        
        total_combinations = np.prod([len(v) for v in param_grid.values()])
        print(f"\n   Total combinations: {total_combinations}")
        print(f"   With 5-fold CV: {total_combinations * 5} fits")
        
        # Grid search
        print("\n🔄 Running Grid Search (this may take a moment)...")
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        print("\n✅ Grid Search Complete!")
        print(f"\n🏆 Best Parameters:")
        for param, value in grid_search.best_params_.items():
            print(f"   {param}: {value}")
        
        print(f"\n📊 Performance:")
        print(f"   Best CV Score: {grid_search.best_score_:.3f}")
        
        # Test performance
        y_pred = grid_search.predict(self.X_test)
        test_accuracy = grid_search.score(self.X_test, self.y_test)
        print(f"   Test Accuracy: {test_accuracy:.3f}")
        
        # Update best pipeline
        self.best_pipeline = {
            'pipeline': grid_search.best_estimator_,
            'metrics': {'accuracy': test_accuracy, 'roc_auc': grid_search.best_score_}
        }
        
        return grid_search
    
    def evaluate_best_pipeline(self):
        """Detailed evaluation of best pipeline."""
        print("\n" + "="*60)
        print("🏆 BEST PIPELINE DETAILED EVALUATION")
        print("="*60)
        
        pipeline = self.best_pipeline['pipeline']
        
        # Make predictions
        y_pred = pipeline.predict(self.X_test)
        y_pred_proba = pipeline.predict_proba(self.X_test)[:, 1]
        
        # Evaluate
        evaluator = ClassificationEvaluator(self.y_test, y_pred, y_pred_proba)
        evaluator.print_evaluation_report()
        
        print("\n📊 Visualizations:")
        evaluator.plot_confusion_matrix()
        evaluator.plot_roc_curve()
    
    def save_pipeline(self, filepath='../models/churn_pipeline.pkl'):
        """
        Save the entire pipeline.
        
        Key benefit: Saves EVERYTHING in one object!
        - Scaler (with fitted parameters)
        - Model (with trained weights)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.best_pipeline['pipeline'], f)
        
        print(f"\n💾 Pipeline saved to: {filepath}")
        print("   Includes: preprocessing + model (everything!)")
        print("\n💡 To use later:")
        print("   with open('churn_pipeline.pkl', 'rb') as f:")
        print("       pipeline = pickle.load(f)")
        print("   prediction = pipeline.predict(new_data)")
    
    def predict_new_customer(self, customer_data):
        """
        Make prediction using the pipeline.
        
        No manual scaling needed - Pipeline handles it!
        """
        if isinstance(customer_data, dict):
            customer_data = pd.DataFrame([customer_data])
        
        pipeline = self.best_pipeline['pipeline']
        
        # Pipeline automatically handles preprocessing!
        prediction = pipeline.predict(customer_data)[0]
        probability = pipeline.predict_proba(customer_data)[0]
        
        print(f"\n🎯 CHURN PREDICTION")
        print(f"   Prediction: {'CHURN' if prediction == 1 else 'NO CHURN'}")
        print(f"   Churn Probability: {probability[1]:.1%}")
        print(f"   Confidence: {'High' if max(probability) > 0.8 else 'Medium' if max(probability) > 0.6 else 'Low'}")
        
        return prediction, probability


def main():
    """Main execution."""
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║   CHURN PREDICTION WITH SKLEARN PIPELINE (The Right Way!)        ║
    ║        Based on DataCamp Supervised Learning Course              ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Initialize
    predictor = ChurnPredictionWithPipeline(test_size=0.2, random_state=42)
    
    # Load data
    df = predictor.load_data()
    
    # Explore
    predictor.explore_data()
    
    # Prepare
    predictor.prepare_data()
    
    # Create pipelines
    predictor.create_pipelines()
    
    # Train and compare
    results = predictor.train_and_compare()
    
    # Optional: Tune hyperparameters
    print("\n" + "="*60)
    response = input("🔍 Run hyperparameter tuning? (y/n): ").lower()
    if response == 'y':
        predictor.tune_best_pipeline()
    
    # Evaluate
    predictor.evaluate_best_pipeline()
    
    # Save
    predictor.save_pipeline()
    
    # Test prediction
    print("\n" + "="*60)
    print("🧪 TESTING PREDICTION")
    print("="*60)
    
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
    
    predictor.predict_new_customer(sample_customer)
    
    print("\n" + "="*60)
    print("✅ COMPLETE!")
    print("="*60)
    print("\n🎓 What's Different from Before:")
    print("   ✅ Using sklearn.pipeline.Pipeline (not custom class)")
    print("   ✅ No manual scaling needed")
    print("   ✅ Safer (prevents data leakage)")
    print("   ✅ Cleaner code")
    print("   ✅ Production-ready")
    print("\n💡 Next Steps:")
    print("   1. Run sklearn_pipeline_guide.py for deep dive")
    print("   2. Experiment with different pipeline configurations")
    print("   3. Deploy this pipeline as an API")


if __name__ == "__main__":
    main()
