import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# PyTorch
import torch
import torch.nn as nn
from torchvision import transforms, models

# Optional SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Vehicle Damage Risk Platform",
    page_icon="🚗",
    layout="wide"
)


# ============================================================
# CONSTANTS
# ============================================================
DAMAGE_CLASSES = [
    "F_Breakage",
    "F_Crushed",
    "F_Normal",
    "R_Breakage",
    "R_Crushed",
    "R_Normal"
]

MODEL_DIR = "saved_models"

DAMAGE_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "damage_classifier.pth"
)

REPAIR_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "repair_cost_model.pkl"
)

SEVERITY_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "severity_model.pkl"
)

SEVERITY_ENCODER_PATH = os.path.join(
    MODEL_DIR,
    "severity_label_encoder.pkl"
)

FRAUD_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "fraud_model.pkl"
)


# ============================================================
# RESNET DAMAGE CLASSIFIER
# ============================================================
class CarClassifierCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.model = models.resnet50(weights=None)

        in_features = self.model.fc.in_features

        self.model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)


# ============================================================
# LOAD MODELS
# ============================================================
@st.cache_resource
def load_models():

    loaded_models = {}

    # ========================================================
    # DAMAGE MODEL
    # ========================================================
    if os.path.exists(DAMAGE_MODEL_PATH):

        try:
            damage_model = CarClassifierCNN()

            checkpoint = torch.load(
                DAMAGE_MODEL_PATH,
                map_location="cpu"
            )

            # ---------------------------------------------
            # Handle multiple checkpoint formats
            # ---------------------------------------------
            if isinstance(checkpoint, dict):

                if "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                else:
                    state_dict = checkpoint

            else:
                state_dict = checkpoint

            # ---------------------------------------------
            # Remove DataParallel prefixes
            # ---------------------------------------------
            cleaned_state_dict = {}

            for k, v in state_dict.items():

                if k.startswith("module."):
                    cleaned_state_dict[k[7:]] = v
                else:
                    cleaned_state_dict[k] = v

            damage_model.load_state_dict(
                cleaned_state_dict,
                strict=False
            )

            damage_model.eval()

            loaded_models["damage_model"] = damage_model

        except Exception as e:

            st.warning(
                f"Could not load damage model: {e}"
            )

            loaded_models["damage_model"] = None

    else:
        loaded_models["damage_model"] = None

    # ========================================================
    # REPAIR MODEL
    # ========================================================
    if os.path.exists(REPAIR_MODEL_PATH):

        try:
            loaded_models["repair_model"] = joblib.load(
                REPAIR_MODEL_PATH
            )

        except Exception as e:

            st.warning(
                f"Could not load repair model: {e}"
            )

            loaded_models["repair_model"] = None

    else:
        loaded_models["repair_model"] = None

    # ========================================================
    # SEVERITY MODEL
    # ========================================================
    if os.path.exists(SEVERITY_MODEL_PATH):

        try:
            loaded_models["severity_model"] = joblib.load(
                SEVERITY_MODEL_PATH
            )

        except Exception as e:

            st.warning(
                f"Could not load severity model: {e}"
            )

            loaded_models["severity_model"] = None

    else:
        loaded_models["severity_model"] = None

    # ========================================================
    # LABEL ENCODER
    # ========================================================
    if os.path.exists(SEVERITY_ENCODER_PATH):

        try:
            loaded_models["severity_encoder"] = joblib.load(
                SEVERITY_ENCODER_PATH
            )

        except Exception as e:

            st.warning(
                f"Could not load severity encoder: {e}"
            )

            loaded_models["severity_encoder"] = None

    else:
        loaded_models["severity_encoder"] = None

    # ========================================================
    # FRAUD MODEL
    # ========================================================
    if os.path.exists(FRAUD_MODEL_PATH):

        try:
            loaded_models["fraud_model"] = joblib.load(
                FRAUD_MODEL_PATH
            )

        except Exception as e:

            st.warning(
                f"Could not load fraud model: {e}"
            )

            loaded_models["fraud_model"] = None

    else:
        loaded_models["fraud_model"] = None

    return loaded_models


models_dict = load_models()


# ============================================================
# DAMAGE CLASSIFICATION
# ============================================================
def predict_damage_class(uploaded_image):

    model = models_dict["damage_model"]

    # Fallback if model unavailable
    if model is None:

        return np.random.choice(DAMAGE_CLASSES)

    image = Image.open(uploaded_image).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    x = transform(image).unsqueeze(0)

    with torch.no_grad():

        outputs = model(x)

        probs = torch.softmax(outputs, dim=1)

        pred_idx = torch.argmax(
            probs,
            dim=1
        ).item()

    return DAMAGE_CLASSES[pred_idx]


# ============================================================
# REPAIR COST
# ============================================================
def predict_repair_cost(
    damage_class,
    vehicle_age,
    mileage
):

    model = models_dict["repair_model"]

    if model is None:

        base_cost = {
            "Front Breakage": 40000,
            "Front Crushed": 90000,
            "Rear Breakage": 35000,
            "Rear Crushed": 80000
        }

        cost = (
            base_cost.get(damage_class, 50000)
            + vehicle_age * 1200
            + mileage * 0.05
        )

        return float(cost)

    X = pd.DataFrame({
        "damage_class": [damage_class],
        "vehicle_age": [vehicle_age],
        "mileage": [mileage]
    })

    prediction = model.predict(X)[0]

    return float(prediction)


# ============================================================
# CLAIM SEVERITY
# ============================================================
def predict_claim_severity(
    damage_class,
    vehicle_age,
    mileage,
    repair_cost
):

    model = models_dict["severity_model"]
    encoder = models_dict["severity_encoder"]

    if model is None or encoder is None:

        if repair_cost < 25000:
            return "Minor"

        elif repair_cost < 60000:
            return "Moderate"

        elif repair_cost < 150000:
            return "Severe"

        else:
            return "Total Loss"

    X = pd.DataFrame({
        "damage_class": [damage_class],
        "vehicle_age": [vehicle_age],
        "mileage": [mileage],
        "repair_cost": [repair_cost]
    })

    pred_encoded = model.predict(X)[0]

    pred_label = encoder.inverse_transform(
        [pred_encoded]
    )[0]

    return pred_label


# ============================================================
# FRAUD PREDICTION
# ============================================================
def predict_fraud(
    damage_class,
    vehicle_age,
    mileage,
    repair_cost,
    claim_amount,
    claim_severity,
    prior_claims,
    policy_age_months,
    days_since_policy_inception
):

    model = models_dict["fraud_model"]

    # ========================================================
    # FALLBACK
    # ========================================================
    if model is None:

        score = 0

        if claim_amount > repair_cost * 1.2:
            score += 2

        if prior_claims >= 3:
            score += 2

        if policy_age_months < 6:
            score += 1

        if claim_severity == "Total Loss":
            score += 1

        probability = min(0.95, score / 8)

        return (
            int(probability > 0.5),
            float(probability),
            None
        )

    # ========================================================
    # MODEL PREDICTION
    # ========================================================
    X = pd.DataFrame({
        "damage_class": [damage_class],
        "vehicle_age": [vehicle_age],
        "mileage": [mileage],
        "repair_cost": [repair_cost],
        "claim_amount": [claim_amount],
        "claim_severity": [claim_severity],
        "prior_claims": [prior_claims],
        "policy_age_months": [policy_age_months],
        "days_since_policy_inception": [
            days_since_policy_inception
        ]
    })

    pred = model.predict(X)[0]

    proba = model.predict_proba(X)[0, 1]

    top_drivers = None

    # ========================================================
    # SHAP EXPLAINABILITY
    # ========================================================
    if SHAP_AVAILABLE:

        try:
            preprocessor = model.named_steps[
                "preprocessor"
            ]

            classifier = model.named_steps[
                "classifier"
            ]

            X_transformed = preprocessor.transform(X)

            feature_names = (
                preprocessor.get_feature_names_out()
            )

            explainer = shap.Explainer(
                classifier.predict_proba,
                X_transformed
            )

            shap_values = explainer(
                X_transformed
            )

            contributions = pd.DataFrame({
                "feature": feature_names,
                "shap_value": shap_values.values[
                    0, :, 1
                ]
            })

            contributions["abs_shap"] = (
                contributions["shap_value"].abs()
            )

            top_drivers = (
                contributions
                .sort_values(
                    "abs_shap",
                    ascending=False
                )
                .head(10)[
                    ["feature", "shap_value"]
                ]
            )

        except Exception as e:

            st.warning(
                f"SHAP explanation failed: {e}"
            )

            top_drivers = None

    return (
        int(pred),
        float(proba),
        top_drivers
    )


# ============================================================
# APP HEADER
# ============================================================
st.title("🚗 Vehicle Damage Risk Platform")

st.markdown("""
Upload a vehicle image and analyze insurance risk using
end-to-end machine learning.

### Features
- Damage Classification
- Repair Cost Estimation
- Claim Severity Prediction
- Fraud Detection
- SHAP Explainability
""")


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("📝 Claim Information")

vehicle_age = st.sidebar.slider(
    "Vehicle Age (Years)",
    min_value=0,
    max_value=20,
    value=5
)

mileage = st.sidebar.number_input(
    "Mileage (km)",
    min_value=0,
    max_value=500000,
    value=60000,
    step=1000
)

claim_amount = st.sidebar.number_input(
    "Claim Amount (₹)",
    min_value=0.0,
    value=120000.0,
    step=1000.0
)

prior_claims = st.sidebar.number_input(
    "Prior Claims",
    min_value=0,
    max_value=20,
    value=1
)

policy_age_months = st.sidebar.number_input(
    "Policy Age (Months)",
    min_value=1,
    max_value=240,
    value=24
)

days_since_policy_inception = (
    st.sidebar.number_input(
        "Days Since Policy Inception",
        min_value=1,
        max_value=10000,
        value=730
    )
)


# ============================================================
# FILE UPLOAD
# ============================================================
uploaded_file = st.file_uploader(
    "📤 Upload Vehicle Image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# MAIN PIPELINE
# ============================================================
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Vehicle Image",
        use_container_width=True
    )

    with st.spinner("Analyzing claim..."):

        damage_class = predict_damage_class(
            uploaded_file
        )

        repair_cost = predict_repair_cost(
            damage_class,
            vehicle_age,
            mileage
        )

        claim_severity = predict_claim_severity(
            damage_class,
            vehicle_age,
            mileage,
            repair_cost
        )

        (
            fraud_flag,
            fraud_probability,
            top_drivers
        ) = predict_fraud(
            damage_class,
            vehicle_age,
            mileage,
            repair_cost,
            claim_amount,
            claim_severity,
            prior_claims,
            policy_age_months,
            days_since_policy_inception
        )

    # ========================================================
    # RESULTS
    # ========================================================
    st.header("📊 Results")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Damage Class",
            damage_class
        )

    with col2:
        st.metric(
            "Repair Cost",
            f"₹ {repair_cost:,.0f}"
        )

    with col3:
        st.metric(
            "Claim Severity",
            claim_severity
        )

    with col4:
        st.metric(
            "Fraud Probability",
            f"{fraud_probability:.2%}"
        )

    # ========================================================
    # FRAUD BANNER
    # ========================================================
    if fraud_probability >= 0.80:

        st.error(
            "🔴 High Fraud Risk — Investigation Recommended"
        )

    elif fraud_probability >= 0.50:

        st.warning(
            "🟠 Medium Fraud Risk — Review Suggested"
        )

    else:

        st.success(
            "🟢 Low Fraud Risk — Claim Appears Legitimate"
        )

    # ========================================================
    # SHAP FACTORS
    # ========================================================
    if top_drivers is not None:

        st.subheader(
            "🔍 Top Fraud Drivers"
        )

        display_df = top_drivers.copy()

        display_df["impact"] = np.where(
            display_df["shap_value"] > 0,
            "Increases Risk",
            "Decreases Risk"
        )

        display_df["importance"] = (
            display_df["shap_value"]
            .abs()
            .round(4)
        )

        st.dataframe(
            display_df[
                [
                    "feature",
                    "impact",
                    "importance"
                ]
            ],
            use_container_width=True
        )

else:

    st.info(
        "Please upload a vehicle image."
    )


# ============================================================
# MODEL STATUS
# ============================================================
with st.expander("🛠 Model Status"):

    status_data = {
        "Model": [
            "Damage Classifier",
            "Repair Cost Model",
            "Claim Severity Model",
            "Severity Encoder",
            "Fraud Detection Model"
        ],

        "Loaded": [
            models_dict["damage_model"] is not None,
            models_dict["repair_model"] is not None,
            models_dict["severity_model"] is not None,
            models_dict["severity_encoder"] is not None,
            models_dict["fraud_model"] is not None
        ]
    }

    status_df = pd.DataFrame(status_data)

    st.dataframe(
        status_df,
        use_container_width=True
    )


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")

st.markdown("""
### 🚀 About This Project

This platform combines:

- 🖼️ PyTorch Computer Vision
- 💰 Repair Cost Estimation
- 📈 Severity Prediction
- 🕵️ Fraud Detection
- 🔍 Explainable AI

### Tech Stack
- Python
- Streamlit
- PyTorch
- Scikit-learn
- XGBoost
- SHAP

Developed by Abhijeet Singh Pawar.
""")
