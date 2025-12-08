"""
Real Data Churn Prediction - Telco Customer Churn from Kaggle
==============================================================
This script works with REAL data, not synthetic!

Dataset: Telco Customer Churn from Kaggle
URL: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

What's different from synthetic data:
1. Missing values to handle
2. Categorical features to encode
3. Class imbalance to address
4. Real business patterns
5. Feature engineering opportunities

Author: Pranav Donepudi
Date: December 2024
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

import warnings

warnings.filterwarnings("ignore")


class RealDataChurnPredictor:
    """
    Complete churn prediction pipeline for REAL Kaggle data.

    Handles all the messiness of real-world data:
    - Missing values
    - Categorical encoding
    - Feature engineering
    - Class imbalance
    """

    def __init__(self, data_path=None, test_size=0.2, random_state=42):
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state
        self.df_original = None
        self.df_processed = None
        self.pipelines = {}
        self.best_pipeline = None

    def load_data(self):
        """
        Load Telco Customer Churn dataset.
        """
        print("\n" + "=" * 70)
        print(" LOADING KAGGLE DATA")
        print("=" * 70)

        # Try multiple possible file locations
        possible_paths = [
            self.data_path,
            "./data/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        ]

        loaded = False
        for path in possible_paths:
            if path and Path(path).exists():
                print(f"Found data: {path}")
                self.df_original = pd.read_csv(path)
                loaded = True
                break

        if not loaded:
            print("\n DATASET NOT FOUND!")
            print("\n Please download the Telco Customer Churn dataset:")
            print(
                "   1. Go to: https://www.kaggle.com/datasets/blastchar/telco-customer-churn"
            )
            print("   2. Click 'Download'")
            print("   3. Save to: churn-prediction-project/data/")
            print("   4. Run this script again")
            print("\nAlternatively, install Kaggle CLI:")
            print("   pip install kaggle")
            print("   kaggle datasets download -d blastchar/telco-customer-churn")
            return None

        print(
            f"\n Dataset loaded: {self.df_original.shape[0]} rows, {self.df_original.shape[1]} columns"
        )
        print(f"   Target variable: Churn")

        return self.df_original

    def explore_data(self):
        """
        Exploratory Data Analysis on real data.
        """
        if self.df_original is None:
            print(" No data loaded!")
            return

        print("\n" + "=" * 70)
        print(" EXPLORATORY DATA ANALYSIS - REAL DATA")
        print("=" * 70)

        # Basic info
        print("\n 1. Dataset Overview:")
        print(f" Shape: {self.df_original.shape}")
        print(
            f" Memory: {self.df_original.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
        )

        # Column types
        print("\n 2. Column Types:")
        print(
            f"   Numeric columns: {self.df_original.select_dtypes(include=[np.number]).columns.tolist()}"
        )
        print(
            f"   Object columns: {self.df_original.select_dtypes(include=['object']).columns.tolist()}"
        )

        # Missing values (REAL DATA HAS THESE!)
        print("\n 3. Missing Values:")
        missing = self.df_original.isnull().sum()
        if missing.sum() == 0:
            print(" No missing values detected")
        else:
            print(missing[missing > 0])

        # Check TotalCharges (known issue in this dataset)
        if "TotalCharges" in self.df_original.columns:
            try:
                pd.to_numeric(self.df_original["TotalCharges"], errors="raise")
                print("TotalCharges is numeric")
            except:
                print(
                    " TotalCharges has non-numeric values (will fix in preprocessing)"
                )

        # Target distribution
        print("\n 4. Target Distribution (Class Balance):")
        if "Churn" in self.df_original.columns:
            churn_counts = self.df_original["Churn"].value_counts()
            churn_pct = self.df_original["Churn"].value_counts(normalize=True) * 100
            print(f"   No Churn: {churn_counts.iloc[0]} ({churn_pct.iloc[0]:.1f}%)")
            print(f"   Churn:    {churn_counts.iloc[1]} ({churn_pct.iloc[1]:.1f}%)")

            if churn_pct.iloc[1] < 40:
                print(" IMBALANCED DATASET - Minority class < 40%")
                print(" Will use class_weight='balanced' in models")

        # Sample data
        print("\n 5. First Few Rows:")
        print(self.df_original.head(3))

        # Key statistics
        print("\n 6. Numerical Features Statistics:")
        numeric_cols = self.df_original.select_dtypes(include=[np.number]).columns
        print(self.df_original[numeric_cols].describe().round(2))

    def preprocess_data(self):
        """
        Clean and preprocess real data.

        Steps:
        1. Handle missing values
        2. Fix data types
        3. Encode categorical features
        4. Feature engineering
        5. Drop unnecessary columns
        """
        print("\n" + "=" * 70)
        print("DATA PREPROCESSING - REAL DATA")
        print("=" * 70)

        if self.df_original is None:
            print("No data loaded!")
            return None

        df = self.df_original.copy()

        # Step 1: Handle TotalCharges (known issue)
        print("\n 1. Fixing TotalCharges column...")
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
            missing_total_charges = df["TotalCharges"].isna().sum()
            if missing_total_charges > 0:
                print(f"   Found {missing_total_charges} missing values")
                print(f"   Filling with median: {df['TotalCharges'].median():.2f}")
                df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

        # Step 2: Drop customerID (not useful for prediction)
        print("\n 2. Dropping customerID...")
        if "customerID" in df.columns:
            df = df.drop("customerID", axis=1)
            print("Dropped customerID")

        # Step 3: Encode target variable
        print("\n 3. Encoding target variable (Churn)...")
        if "Churn" in df.columns:
            df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
            print("   Yes → 1, No → 0")

        # Step 4: Encode binary categorical features
        print("\n 4. Encoding binary features...")
        binary_maps = {
            "gender": {"Male": 1, "Female": 0},
            "Partner": {"Yes": 1, "No": 0},
            "Dependents": {"Yes": 1, "No": 0},
            "PhoneService": {"Yes": 1, "No": 0},
            "PaperlessBilling": {"Yes": 1, "No": 0},
        }

        for col, mapping in binary_maps.items():
            if col in df.columns:
                df[col] = df[col].map(mapping)
                print(f"   {col}: {mapping}")

        # Step 5: One-hot encode multi-category features
        print("\n 5. One-hot encoding categorical features...")
        categorical_cols = []

        # Identify remaining categorical columns
        for col in df.columns:
            if df[col].dtype == "object" and col != "Churn":
                categorical_cols.append(col)

        if categorical_cols:
            print(f" Encoding: {categorical_cols}")
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
            print(f" Created {len(df.columns)} total columns after encoding")

        # Step 6: Feature Engineering (Optional but recommended!)
        print("\n 6. Feature Engineering...")

        # Tenure groups
        if "tenure" in df.columns:
            df["tenure_group"] = pd.cut(
                df["tenure"],
                bins=[-1, 12, 24, 48, 100],
                labels=[0, 1, 2, 3],
                include_lowest=True,
            )
            df["tenure_group"] = df["tenure_group"].astype(int)
            print(" Created tenure_group (0-1yr, 1-2yr, 2-4yr, 4+yr)")

        # Charges ratio
        if "MonthlyCharges" in df.columns and "TotalCharges" in df.columns:
            df["charges_ratio"] = df["MonthlyCharges"] / (df["TotalCharges"] + 1)
            print(" Created charges_ratio (Monthly/Total)")

        # Tenure per dollar
        if "tenure" in df.columns and "MonthlyCharges" in df.columns:
            df["tenure_per_charge"] = df["tenure"] / (df["MonthlyCharges"] + 1)
            print(" Created tenure_per_charge")

        self.df_processed = df

        print("\n" + "=" * 70)
        print("PREPROCESSING COMPLETE")
        print("=" * 70)
        print(f"Final shape: {df.shape}")
        print(f"Features: {df.columns.tolist()}")

        return df

    def prepare_for_modeling(self):
        """
        Split data into train/test sets.
        """
        print("\n" + "=" * 70)
        print("TRAIN-TEST SPLIT")
        print("=" * 70)

        if self.df_processed is None:
            print("Data not preprocessed yet!")
            return

        # Separate features and target
        X = self.df_processed.drop("Churn", axis=1)
        y = self.df_processed["Churn"]

        # Train-test split with stratification (important for imbalanced data!)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        print(f"\nTrain set: {len(self.X_train)} samples")
        print(f"Test set:  {len(self.X_test)} samples")
        print(f"\nTrain churn rate: {self.y_train.mean() * 100:.1f}%")
        print(f"Test churn rate:  {self.y_test.mean() * 100:.1f}%")
        print("\n Stratification maintained class balance!")

    def create_pipelines(self):
        """
        Create pipelines with class_weight for imbalanced data.
        """
        print("\n" + "=" * 70)
        print("CREATING PIPELINES FOR IMBALANCED DATA")
        print("=" * 70)

        # Pipeline 1: Logistic Regression (with class balancing)
        pipe_lr = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        random_state=self.random_state,
                        max_iter=1000,
                        class_weight="balanced",  # ← Handles class imbalance!
                    ),
                ),
            ]
        )

        # Pipeline 2: Random Forest (with class balancing)
        pipe_rf = Pipeline(
            [
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=100,
                        max_depth=10,
                        random_state=self.random_state,
                        class_weight="balanced",  # ← Handles class imbalance!
                    ),
                )
            ]
        )

        # Pipeline 3: Decision Tree
        pipe_dt = Pipeline(
            [
                (
                    "classifier",
                    DecisionTreeClassifier(
                        max_depth=8,
                        random_state=self.random_state,
                        class_weight="balanced",
                    ),
                )
            ]
        )

        self.pipelines = {
            "Logistic Regression": pipe_lr,
            "Random Forest": pipe_rf,
            "Decision Tree": pipe_dt,
        }

        print("\n Created 3 pipelines with class_weight='balanced'")
        print("   This gives more weight to minority class (churners)")

        return self.pipelines

    def train_and_evaluate(self):
        """
        Train all pipelines and compare.
        """
        print("\n" + "=" * 70)
        print(" TRAINING MODELS ON REAL DATA")
        print("=" * 70)

        results = []

        for name, pipeline in self.pipelines.items():
            print(f"\n Training {name}...")

            # Train
            pipeline.fit(self.X_train, self.y_train)

            # Predict
            y_pred = pipeline.predict(self.X_test)
            y_pred_proba = pipeline.predict_proba(self.X_test)[:, 1]

            # Evaluate - RECALL IS CRITICAL FOR CHURN!
            metrics = {
                "accuracy": accuracy_score(self.y_test, y_pred),
                "precision": precision_score(self.y_test, y_pred),
                "recall": recall_score(self.y_test, y_pred),  # ← Most important!
                "f1": f1_score(self.y_test, y_pred),
                "roc_auc": roc_auc_score(self.y_test, y_pred_proba),
            }

            # Cross-validation
            cv_scores = cross_val_score(
                pipeline, self.X_train, self.y_train, cv=5, scoring="recall"
            )

            results.append(
                {
                    "Model": name,
                    "Accuracy": f"{metrics['accuracy']:.3f}",
                    "Precision": f"{metrics['precision']:.3f}",
                    "Recall": f"{metrics['recall']:.3f}",  # ← Focus on this!
                    "F1-Score": f"{metrics['f1']:.3f}",
                    "ROC-AUC": f"{metrics['roc_auc']:.3f}",
                    "CV Recall": f"{cv_scores.mean():.3f} ± {cv_scores.std():.3f}",
                }
            )

            self.pipelines[name] = {"pipeline": pipeline, "metrics": metrics}

            print(
                f" Recall: {metrics['recall']:.3f} (catching {metrics['recall'] * 100:.1f}% of churners)"
            )

        # Display results
        print("\n" + "=" * 70)
        print(" MODEL COMPARISON - FOCUS ON RECALL!")
        print("=" * 70)
        results_df = pd.DataFrame(results)
        print("\n" + results_df.to_string(index=False))

        # Select best by recall (most important for churn!)
        best_name = max(
            self.pipelines.items(), key=lambda x: x[1]["metrics"]["recall"]
        )[0]
        self.best_pipeline = self.pipelines[best_name]

        print(f"\n Best Model (by Recall): {best_name}")
        print(f"   Recall: {self.best_pipeline['metrics']['recall']:.3f}")
        print(
            f"   This catches {self.best_pipeline['metrics']['recall'] * 100:.1f}% of churners!"
        )

        # Show confusion matrix
        pipeline = self.best_pipeline["pipeline"]
        y_pred = pipeline.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)

        print("\n Confusion Matrix:")
        print(f"   True Negatives:  {cm[0][0]} (correctly predicted no churn)")
        print(f"   False Positives: {cm[0][1]} (false alarms)")
        print(f"   False Negatives: {cm[1][0]} (MISSED CHURNERS - bad!)")
        print(f"   True Positives:  {cm[1][1]} (correctly caught churners)")

        return results_df

    def analyze_feature_importance(self):
        """
        Show which features matter most (for tree-based models).
        """
        print("\n" + "=" * 70)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("=" * 70)

        # Get Random Forest pipeline
        rf_pipeline = self.pipelines.get("Random Forest", {}).get("pipeline")

        if rf_pipeline:
            # Get feature importance
            rf_model = rf_pipeline.named_steps["classifier"]
            importances = rf_model.feature_importances_
            feature_names = self.X_train.columns

            # Create DataFrame
            fi_df = pd.DataFrame(
                {"Feature": feature_names, "Importance": importances}
            ).sort_values("Importance", ascending=False)

            print("\nTop 15 Most Important Features:")
            print(fi_df.head(15).to_string(index=False))

            # Plot
            plt.figure(figsize=(10, 8))
            top_features = fi_df.head(15)
            plt.barh(range(len(top_features)), top_features["Importance"])
            plt.yticks(range(len(top_features)), top_features["Feature"])
            plt.xlabel("Importance")
            plt.title("Top 15 Features for Churn Prediction")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(
                "./outputs/feature_importance.png", dpi=300, bbox_inches="tight"
            )
            plt.show()

            print("\n  Key Insights:")
            print("   Common important features in Telco dataset:")
            print("   - Contract type (month-to-month = high churn)")
            print("   - Tenure (new customers churn more)")
            print("   - Monthly charges (higher = more churn)")
            print("   - Internet service type")
            print("   - Payment method")

    def save_pipeline(self, filepath="./models/churn_pipeline_real_data.pkl"):
        """
        Save the best pipeline.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "wb") as f:
            pickle.dump(self.best_pipeline["pipeline"], f)

        print(f"\n Best pipeline saved to: {filepath}")
        print("  Trained on REAL Kaggle data")
        print("  Ready for deployment!")


def main():
    """
    Main execution.
    """
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║         REAL DATA CHURN PREDICTION                               ║
    ║      Using Telco Customer Churn from Kaggle                      ║
    ║                                                                  ║
    ║  This is REAL ML - handling real-world data challenges!         ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)

    # Initialize
    predictor = RealDataChurnPredictor()

    # Load data
    df = predictor.load_data()
    if df is None:
        return  # Exit if data not found

    # Explore
    predictor.explore_data()

    # Preprocess
    predictor.preprocess_data()

    # Prepare for modeling
    predictor.prepare_for_modeling()

    # Create pipelines
    predictor.create_pipelines()

    # Train and evaluate
    results = predictor.train_and_evaluate()

    # Analyze features
    predictor.analyze_feature_importance()

    # Save
    predictor.save_pipeline()

    print("\n" + "=" * 70)
    print("✅ COMPLETE - REAL DATA EXPERIENCE!")
    print("=" * 70)
    print("\n🎓 What You Learned:")
    print("   ✓ Handling missing values in real data")
    print("   ✓ Encoding categorical features")
    print("   ✓ Dealing with class imbalance")
    print("   ✓ Feature engineering")
    print("   ✓ Interpreting results in business context")
    print("   ✓ Building portfolio-worthy projects")
    print("\n🚀 This is production-level ML code!")


if __name__ == "__main__":
    main()
