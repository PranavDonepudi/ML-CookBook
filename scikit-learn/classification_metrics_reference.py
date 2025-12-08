"""
Classification Metrics Reference - Scikit-learn
================================================
Based on: DataCamp Supervised Learning Course - Chapter 1

This file contains working examples of all classification concepts.
Use this as your go-to reference when building classification models.

Last Updated: December 2025
Author: Pranav Donepudi
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns


class ClassificationEvaluator:
    """
    Complete evaluation toolkit for classification models.

    Key Concepts from DataCamp Course:
    1. Train-test split: Prevents overfitting evaluation
    2. Accuracy: Overall correctness (use when classes balanced)
    3. Precision: Of positive predictions, how many correct? (minimize false positives)
    4. Recall: Of actual positives, how many found? (minimize false negatives)
    5. F1-Score: Harmonic mean of precision and recall
    6. Confusion Matrix: Shows all prediction outcomes
    """

    def __init__(self, y_true, y_pred, y_pred_proba=None):
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba

    def calculate_all_metrics(self):
        """
        Calculate all classification metrics.

        Returns:
            dict: Dictionary containing all metrics
        """
        metrics = {
            "accuracy": accuracy_score(self.y_true, self.y_pred),
            "precision": precision_score(
                self.y_true, self.y_pred, average="binary", zero_division=0
            ),
            "recall": recall_score(
                self.y_true, self.y_pred, average="binary", zero_division=0
            ),
            "f1_score": f1_score(
                self.y_true, self.y_pred, average="binary", zero_division=0
            ),
        }

        if self.y_pred_proba is not None:
            metrics["roc_auc"] = roc_auc_score(self.y_true, self.y_pred_proba)

        return metrics

    def print_evaluation_report(self):
        """
        Print comprehensive evaluation report.
        """
        print("=" * 60)
        print("CLASSIFICATION MODEL EVALUATION REPORT")
        print("=" * 60)

        metrics = self.calculate_all_metrics()

        print("\n📊 CORE METRICS:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")

        if "roc_auc" in metrics:
            print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")

        print("\n🎯 CONFUSION MATRIX:")
        cm = confusion_matrix(self.y_true, self.y_pred)
        print(f"  True Negatives:  {cm[0][0]}")
        print(f"  False Positives: {cm[0][1]}")
        print(f"  False Negatives: {cm[1][0]}")
        print(f"  True Positives:  {cm[1][1]}")

        print("\n📋 DETAILED CLASSIFICATION REPORT:")
        print(classification_report(self.y_true, self.y_pred))

        print("\n💡 INTERPRETATION GUIDE:")
        self._print_interpretation(metrics, cm)

    def _print_interpretation(self, metrics, cm):
        """
        Provide interpretation of the metrics.
        """
        # Calculate rates
        total = cm.sum()
        tn, fp, fn, tp = cm.ravel()

        print(
            f"  • Model correctly classified {metrics['accuracy'] * 100:.1f}% of samples"
        )
        print(f"  • Of {tp + fp} positive predictions, {tp} were correct (Precision)")
        print(f"  • Of {tp + fn} actual positives, {tp} were found (Recall)")

        # Business context
        print("\n  Business Context:")
        if metrics["precision"] > 0.8:
            print(
                "  ✓ High precision: Few false alarms (good for costly interventions)"
            )
        else:
            print("  ⚠ Lower precision: Many false positives (review threshold)")

        if metrics["recall"] > 0.8:
            print(
                "  ✓ High recall: Catching most positive cases (good for critical detection)"
            )
        else:
            print(
                "  ⚠ Lower recall: Missing some positive cases (consider feature engineering)"
            )

    def plot_confusion_matrix(self, save_path=None):
        """
        Visualize confusion matrix.
        """
        cm = confusion_matrix(self.y_true, self.y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No Churn", "Churn"],
            yticklabels=["No Churn", "Churn"],
        )
        plt.title("Confusion Matrix")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def plot_roc_curve(self, save_path=None):
        """
        Plot ROC curve if probabilities available.
        """
        if self.y_pred_proba is None:
            print("⚠ Cannot plot ROC curve: probability predictions not provided")
            return

        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_pred_proba)
        roc_auc = roc_auc_score(self.y_true, self.y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(
            fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})"
        )
        plt.plot(
            [0, 1],
            [0, 1],
            color="navy",
            lw=2,
            linestyle="--",
            label="Random Classifier",
        )
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()


def demonstrate_train_test_split():
    """
    Demonstrate the importance of train-test split.

    Key Concept: Never evaluate on training data - leads to overfitting!
    """
    print("\n" + "=" * 60)
    print("TRAIN-TEST SPLIT DEMONSTRATION")
    print("=" * 60)

    # Create sample data
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42
    )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n📦 Dataset Split:")
    print(f"  Total samples: {len(X)}")
    print(
        f"  Training set: {len(X_train)} samples ({len(X_train) / len(X) * 100:.1f}%)"
    )
    print(f"  Test set: {len(X_test)} samples ({len(X_test) / len(X) * 100:.1f}%)")

    print(f"\n⚖️ Class Distribution:")
    print(f"  Training - Class 0: {sum(y_train == 0)}, Class 1: {sum(y_train == 1)}")
    print(f"  Test - Class 0: {sum(y_test == 0)}, Class 1: {sum(y_test == 1)}")

    print("\n💡 Key Points:")
    print("  • test_size=0.2 means 80% train, 20% test (common split)")
    print("  • random_state=42 ensures reproducibility")
    print("  • stratify=y maintains class balance in both sets")

    return X_train, X_test, y_train, y_test


def demonstrate_model_complexity_tradeoff():
    """
    Demonstrate relationship between model complexity and performance.

    Key Concept: More complex models can overfit!
    """
    print("\n" + "=" * 60)
    print("MODEL COMPLEXITY vs PERFORMANCE")
    print("=" * 60)

    from sklearn.datasets import make_classification
    from sklearn.tree import DecisionTreeClassifier

    # Create data
    X, y = make_classification(
        n_samples=500, n_features=10, n_informative=8, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Test different complexities
    max_depths = [1, 2, 5, 10, 20, None]
    train_scores = []
    test_scores = []

    print("\n📈 Testing different model complexities (Decision Tree max_depth):\n")

    for depth in max_depths:
        dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
        dt.fit(X_train, y_train)

        train_score = dt.score(X_train, y_train)
        test_score = dt.score(X_test, y_test)

        train_scores.append(train_score)
        test_scores.append(test_score)

        depth_str = f"max_depth={depth}" if depth else "max_depth=None (unlimited)"
        print(f"  {depth_str:30} → Train: {train_score:.3f}, Test: {test_score:.3f}")

    print("\n💡 Observations:")
    print("  • Simple models (depth=1,2): Underfitting - low train & test scores")
    print(
        "  • Complex models (depth=None): Overfitting - high train, lower test scores"
    )
    print("  • Optimal complexity: Best test score with reasonable train score")
    print("  • Gap between train/test scores indicates overfitting")

    # Plot
    plt.figure(figsize=(10, 6))
    x_labels = [str(d) if d else "None" for d in max_depths]
    x_pos = range(len(x_labels))

    plt.plot(x_pos, train_scores, "o-", label="Training Accuracy", linewidth=2)
    plt.plot(x_pos, test_scores, "s-", label="Test Accuracy", linewidth=2)
    plt.xticks(x_pos, x_labels)
    plt.xlabel("Model Complexity (max_depth)")
    plt.ylabel("Accuracy")
    plt.title("Model Complexity vs Performance")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# =============================================================================
# COMPLETE EXAMPLE: Churn Prediction Pipeline
# =============================================================================


def complete_churn_prediction_example():
    """
    Complete example: Train and evaluate a churn prediction model.

    This demonstrates everything from Chapter 1 of the DataCamp course.
    """
    print("\n" + "=" * 60)
    print("COMPLETE CHURN PREDICTION EXAMPLE")
    print("=" * 60)

    # Step 1: Create synthetic churn data
    from sklearn.datasets import make_classification

    print("\n📊 Step 1: Loading Data")
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_classes=2,
        weights=[0.7, 0.3],  # 30% churn rate (realistic)
        random_state=42,
    )
    print(f"  Dataset: {X.shape[0]} customers, {X.shape[1]} features")
    print(f"  Churn rate: {sum(y) / len(y) * 100:.1f}%")

    # Step 2: Train-test split
    print("\n🔀 Step 2: Splitting Data")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Training set: {len(X_train)} samples")
    print(f"  Test set: {len(X_test)} samples")

    # Step 3: Train model
    print("\n🤖 Step 3: Training Model")
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    print("  ✓ Logistic Regression model trained")

    # Step 4: Make predictions
    print("\n🎯 Step 4: Making Predictions")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    print(f"  Predicted {sum(y_pred)} churners out of {len(y_pred)} customers")

    # Step 5: Evaluate
    print("\n📊 Step 5: Evaluating Model")
    evaluator = ClassificationEvaluator(y_test, y_pred, y_pred_proba)
    evaluator.print_evaluation_report()

    # Step 6: Visualize
    print("\n📈 Step 6: Visualizations")
    evaluator.plot_confusion_matrix()
    evaluator.plot_roc_curve()

    return model, evaluator


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║  CLASSIFICATION METRICS REFERENCE - SCIKIT-LEARN                 ║
    ║  Based on: DataCamp Supervised Learning Course                   ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)

    # Run demonstrations
    print("\n🎓 Running Educational Demonstrations...\n")

    # Demo 1: Train-test split
    X_train, X_test, y_train, y_test = demonstrate_train_test_split()

    # Demo 2: Model complexity
    demonstrate_model_complexity_tradeoff()

    # Demo 3: Complete pipeline
    model, evaluator = complete_churn_prediction_example()

    print("\n" + "=" * 60)
    print("✅ ALL DEMONSTRATIONS COMPLETE!")
    print("=" * 60)
    print("\n💡 Next Steps:")
    print("  1. Modify this code to use real churn dataset")
    print("  2. Try different models (Random Forest, XGBoost)")
    print("  3. Implement feature engineering")
    print("  4. Deploy as API using Flask/FastAPI")
    print("\n📁 Save this file as your reference for future projects!")
