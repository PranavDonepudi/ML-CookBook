"""
FastAPI Deployment for Churn Prediction Model
==============================================
Production-ready API with:
- Model loading
- Input validation
- Prediction endpoint
- Health checks
- Error handling
- API documentation

Author: Pranav Donepudi
Date: December 2025
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, List
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import uvicorn

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_PATH = Path("../models/best_churn_model.pkl")
METADATA_PATH = Path("../models/model_metadata.txt")


# =============================================================================
# PYDANTIC MODELS (Input/Output Validation)
# =============================================================================


class CustomerInput(BaseModel):
    """
    Input schema for customer data.
    All fields match the Telco Customer Churn dataset.
    """

    # Demographics
    gender: str = Field(..., description="Male or Female")
    SeniorCitizen: int = Field(..., ge=0, le=1, description="0 or 1")
    Partner: str = Field(..., description="Yes or No")
    Dependents: str = Field(..., description="Yes or No")

    # Account Info
    tenure: int = Field(..., ge=0, le=72, description="Months with company (0-72)")
    PhoneService: str = Field(..., description="Yes or No")
    MultipleLines: str = Field(..., description="Yes, No, or No phone service")

    # Services
    InternetService: str = Field(..., description="DSL, Fiber optic, or No")
    OnlineSecurity: str = Field(..., description="Yes, No, or No internet service")
    OnlineBackup: str = Field(..., description="Yes, No, or No internet service")
    DeviceProtection: str = Field(..., description="Yes, No, or No internet service")
    TechSupport: str = Field(..., description="Yes, No, or No internet service")
    StreamingTV: str = Field(..., description="Yes, No, or No internet service")
    StreamingMovies: str = Field(..., description="Yes, No, or No internet service")

    # Contract
    Contract: str = Field(..., description="Month-to-month, One year, or Two year")
    PaperlessBilling: str = Field(..., description="Yes or No")
    PaymentMethod: str = Field(
        ...,
        description="Electronic check, Mailed check, Bank transfer (automatic), or Credit card (automatic)",
    )

    # Charges
    MonthlyCharges: float = Field(..., ge=0, description="Monthly charges in dollars")
    TotalCharges: float = Field(..., ge=0, description="Total charges in dollars")

    class Config:
        schema_extra = {
            "example": {
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 85.0,
                "TotalCharges": 1020.0,
            }
        }


class PredictionOutput(BaseModel):
    """Output schema for predictions."""

    churn_prediction: str = Field(..., description="Yes or No")
    churn_probability: float = Field(
        ..., ge=0, le=1, description="Probability of churn (0-1)"
    )
    risk_level: str = Field(..., description="Low, Medium, or High")
    confidence: float = Field(..., ge=0, le=1, description="Model confidence")

    # Business insights
    key_risk_factors: List[str] = Field(
        ..., description="Top factors contributing to churn risk"
    )
    recommendation: str = Field(..., description="Recommended action")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    model_info: Dict


# =============================================================================
# DATA PREPROCESSING & FEATURE ENGINEERING
# =============================================================================


class ChurnPreprocessor:
    """
    Handles all preprocessing and feature engineering.
    Must match the training pipeline exactly!
    """

    def __init__(self):
        self.feature_names = None

    def align_with_training_features(
        self, df: pd.DataFrame, expected_features
    ) -> pd.DataFrame:
        """
        Ensure DataFrame has exactly the same columns as training data.
        Add missing columns with zeros, drop extra columns.
        """
        if expected_features is None:
            return df

        # Add missing columns with zeros
        for col in expected_features:
            if col not in df.columns:
                df[col] = 0

        # Keep only expected columns in the same order
        df = df[expected_features]

        return df

    def preprocess(self, customer_data: Dict) -> pd.DataFrame:
        """
        Convert raw customer data to model-ready features.

        CRITICAL: Must match EXACTLY how training data was preprocessed!

        Steps:
        1. Convert to DataFrame
        2. Validate and clean input data
        3. Encode binary features
        4. One-hot encode categorical features (drop_first=True to match training!)
        5. Engineer new features
        6. Ensure all columns match training data
        """
        # Step 1: Convert to DataFrame
        df = pd.DataFrame([customer_data])

        # Step 2: Validate and clean numeric fields
        numeric_fields = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
        for field in numeric_fields:
            if field in df.columns:
                # Convert to numeric, handle any issues
                df[field] = pd.to_numeric(df[field], errors="coerce")
                # Fill NaN with sensible defaults
                if field == "tenure":
                    df[field] = df[field].fillna(0)
                elif field in ["MonthlyCharges", "TotalCharges"]:
                    df[field] = df[field].fillna(0)
                elif field == "SeniorCitizen":
                    df[field] = df[field].fillna(0)

        # Step 3: Encode binary features
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
                # If mapping failed, default to 0
                df[col] = df[col].fillna(0)

        # Step 4: One-hot encode categorical features
        # IMPORTANT: Must match training exactly - these are the categorical columns from raw data
        categorical_cols = []
        for col in df.columns:
            if df[col].dtype == "object":
                categorical_cols.append(col)

        # Use drop_first=True to match training
        if categorical_cols:
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

        # Step 5: Feature Engineering (must match training!)
        df = self.engineer_features(df)

        # Step 6: Final validation - check for NaN
        if df.isnull().any().any():
            # Log which columns have NaN
            nan_cols = df.columns[df.isnull().any()].tolist()
            print(f"Warning: NaN found in columns: {nan_cols}")
            # Fill all remaining NaN with 0
            df = df.fillna(0)

        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the EXACT same feature engineering as training.
        CRITICAL: Handle all NaN cases to prevent model errors!
        """
        df_eng = df.copy()

        # Ensure numeric columns are actually numeric
        numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
        for col in numeric_cols:
            if col in df_eng.columns:
                df_eng[col] = pd.to_numeric(df_eng[col], errors="coerce")

        # Fill any NaN in base features with 0
        df_eng["tenure"] = df_eng["tenure"].fillna(0)
        df_eng["MonthlyCharges"] = df_eng["MonthlyCharges"].fillna(0)
        df_eng["TotalCharges"] = df_eng["TotalCharges"].fillna(0)
        df_eng["SeniorCitizen"] = df_eng["SeniorCitizen"].fillna(0)

        # Aggregation - Count services not subscribed
        service_no_cols = [
            col
            for col in df_eng.columns
            if any(
                x in col
                for x in [
                    "OnlineSecurity_No",
                    "OnlineBackup_No",
                    "DeviceProtection_No",
                    "TechSupport_No",
                    "StreamingTV_No",
                    "StreamingMovies_No",
                ]
            )
        ]

        if service_no_cols:
            df_eng["services_not_subscribed"] = df_eng[service_no_cols].sum(axis=1)
            df_eng["total_services"] = 6 - df_eng["services_not_subscribed"]
        else:
            df_eng["total_services"] = 3

        # Ratio features - use fillna to handle division edge cases
        df_eng["monthly_total_ratio"] = (
            df_eng["MonthlyCharges"] / (df_eng["TotalCharges"] + 1)
        ).fillna(0)
        df_eng["services_per_dollar"] = (
            df_eng["total_services"] / (df_eng["MonthlyCharges"] + 1)
        ).fillna(0)
        df_eng["tenure_per_dollar"] = (
            df_eng["tenure"] / (df_eng["MonthlyCharges"] + 1)
        ).fillna(0)

        # Interaction features
        df_eng["tenure_x_charges"] = (
            df_eng["tenure"] * df_eng["MonthlyCharges"]
        ).fillna(0)
        df_eng["senior_charges"] = (
            df_eng["SeniorCitizen"] * df_eng["MonthlyCharges"]
        ).fillna(0)

        # Binning features - handle out of range values
        try:
            df_eng["tenure_group"] = pd.cut(
                df_eng["tenure"],
                bins=[-1, 12, 24, 48, 100],
                labels=[0, 1, 2, 3],
                include_lowest=True,
            ).astype(int)
        except:
            # If binning fails, use a simple approach
            df_eng["tenure_group"] = 0
            df_eng.loc[df_eng["tenure"] > 12, "tenure_group"] = 1
            df_eng.loc[df_eng["tenure"] > 24, "tenure_group"] = 2
            df_eng.loc[df_eng["tenure"] > 48, "tenure_group"] = 3

        try:
            df_eng["charge_level"] = pd.cut(
                df_eng["MonthlyCharges"],
                bins=[0, 35, 70, 150],
                labels=[0, 1, 2],
                include_lowest=True,
            ).astype(int)
        except:
            # If binning fails, use a simple approach
            df_eng["charge_level"] = 0
            df_eng.loc[df_eng["MonthlyCharges"] > 35, "charge_level"] = 1
            df_eng.loc[df_eng["MonthlyCharges"] > 70, "charge_level"] = 2

        # Boolean flags
        df_eng["high_risk_new"] = (
            (df_eng["tenure"] < 12) & (df_eng["MonthlyCharges"] > 70)
        ).astype(int)
        df_eng["low_engagement"] = (df_eng["total_services"] < 2).astype(int)
        df_eng["long_term"] = (df_eng["tenure"] > 36).astype(int)

        # Calculate high value (approximate)
        df_eng["high_value"] = (df_eng["TotalCharges"] > 3500).astype(int)

        # Time-based features
        df_eng["avg_monthly_charges"] = (
            df_eng["TotalCharges"] / (df_eng["tenure"] + 1)
        ).fillna(0)
        df_eng["customer_ltv"] = (df_eng["tenure"] * df_eng["MonthlyCharges"]).fillna(0)

        # Risk score
        df_eng["risk_score"] = 0
        df_eng["risk_score"] += (df_eng["tenure"] < 12).astype(int) * 3
        df_eng["risk_score"] += (df_eng["MonthlyCharges"] > 70).astype(int) * 2
        df_eng["risk_score"] += (df_eng["total_services"] < 2).astype(int) * 2

        if "Contract_Month-to-month" in df_eng.columns:
            df_eng["risk_score"] += df_eng["Contract_Month-to-month"].fillna(0) * 4

        # Final check: Replace any remaining NaN with 0
        df_eng = df_eng.fillna(0)

        return df_eng

    def identify_risk_factors(
        self, customer_data: Dict, probability: float
    ) -> List[str]:
        """
        Identify key risk factors for this customer.
        """
        risk_factors = []

        # Check various risk indicators
        if customer_data["tenure"] < 12:
            risk_factors.append("New customer (tenure < 12 months)")

        if customer_data["Contract"] == "Month-to-month":
            risk_factors.append("Month-to-month contract (high churn risk)")

        if customer_data["MonthlyCharges"] > 70:
            risk_factors.append("High monthly charges (> $70)")

        # Count services
        services = [
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]
        service_count = sum(1 for s in services if customer_data.get(s) == "Yes")

        if service_count < 2:
            risk_factors.append("Low engagement (< 2 services)")

        if customer_data["InternetService"] == "Fiber optic":
            risk_factors.append("Fiber optic service (higher churn rates)")

        if customer_data["PaymentMethod"] == "Electronic check":
            risk_factors.append("Electronic check payment (less stable)")

        if not risk_factors:
            risk_factors.append("No major risk factors identified")

        return risk_factors[:3]  # Return top 3


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

# Initialize FastAPI
app = FastAPI(
    title="Churn Prediction API",
    description="Predict customer churn with 93.9% recall rate",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Global variables
model = None
preprocessor = ChurnPreprocessor()
model_metadata = {}
expected_features = None  # Will store the exact features model expects


# =============================================================================
# STARTUP & SHUTDOWN
# =============================================================================


@app.on_event("startup")
async def load_model():
    """Load model on startup."""
    global model, model_metadata, expected_features

    try:
        # Load model
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)

        print(f"✅ Model loaded from {MODEL_PATH}")

        # Extract expected feature names from the model
        # For Pipeline, we need to get the feature names from the last step
        if hasattr(model, "named_steps"):
            # It's a Pipeline
            if hasattr(model.named_steps["classifier"], "feature_names_in_"):
                expected_features = model.named_steps["classifier"].feature_names_in_
            elif hasattr(model, "feature_names_in_"):
                expected_features = model.feature_names_in_
        elif hasattr(model, "feature_names_in_"):
            expected_features = model.feature_names_in_

        if expected_features is not None:
            print(f"✅ Model expects {len(expected_features)} features")
        else:
            print("⚠️  Warning: Could not extract expected feature names from model")

        # Load metadata
        if METADATA_PATH.exists():
            with open(METADATA_PATH, "r") as f:
                for line in f:
                    if ":" in line:
                        key, value = line.strip().split(":", 1)
                        model_metadata[key.strip()] = value.strip()

        print(f"✅ Model metadata loaded")
        print(f"   Best approach: {model_metadata.get('best_approach', 'Unknown')}")
        print(f"   Recall: {model_metadata.get('recall', 'Unknown')}")

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    print("🛑 Shutting down API...")


# =============================================================================
# API ENDPOINTS
# =============================================================================


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "Churn Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    Returns API status and model information.
    """
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "model_info": model_metadata,
    }


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
async def predict_churn(customer: CustomerInput):
    """
    Predict churn for a customer.

    **Input**: Customer data (19 fields)

    **Output**:
    - Churn prediction (Yes/No)
    - Churn probability (0-1)
    - Risk level (Low/Medium/High)
    - Key risk factors
    - Recommended action

    **Example**:
    ```json
    {
        "gender": "Male",
        "SeniorCitizen": 0,
        "tenure": 12,
        "MonthlyCharges": 85.0,
        ...
    }
    ```
    """
    try:
        # Check if model is loaded
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        # Convert input to dict
        customer_dict = customer.dict()

        # Preprocess
        X = preprocessor.preprocess(customer_dict)

        # Align columns with training data
        X = preprocessor.align_with_training_features(X, expected_features)

        # Predict
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0][1]  # Probability of churn

        # Determine risk level
        if probability >= 0.7:
            risk_level = "High"
        elif probability >= 0.4:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        # Get risk factors
        risk_factors = preprocessor.identify_risk_factors(customer_dict, probability)

        # Recommendation
        if prediction == 1:
            if probability >= 0.8:
                recommendation = (
                    "URGENT: Contact customer immediately with retention offer"
                )
            elif probability >= 0.6:
                recommendation = "High priority: Schedule call within 48 hours"
            else:
                recommendation = "Medium priority: Send personalized retention email"
        else:
            recommendation = "Low risk: Continue normal engagement"

        # Confidence (distance from decision boundary)
        confidence = abs(probability - 0.5) * 2

        return {
            "churn_prediction": "Yes" if prediction == 1 else "No",
            "churn_probability": round(probability, 3),
            "risk_level": risk_level,
            "confidence": round(confidence, 3),
            "key_risk_factors": risk_factors,
            "recommendation": recommendation,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(customers: List[CustomerInput]):
    """
    Predict churn for multiple customers.

    **Input**: List of customer data

    **Output**: List of predictions

    **Limit**: Max 100 customers per request
    """
    if len(customers) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 customers per batch")

    try:
        results = []
        errors = []

        for i, customer in enumerate(customers):
            try:
                result = await predict_churn(customer)
                results.append(result)
            except Exception as e:
                errors.append({"customer_index": i, "error": str(e)})

        response = {
            "count": len(results),
            "successful": len(results),
            "failed": len(errors),
            "predictions": results,
        }

        if errors:
            response["errors"] = errors

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


# =============================================================================
# RUN SERVER
# =============================================================================

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║         CHURN PREDICTION API                                     ║
    ║         Starting FastAPI server...                               ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)

    # Run with uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info",
    )
