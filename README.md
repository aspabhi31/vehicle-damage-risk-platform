#  Vehicle Damage Risk Platform

An end-to-end AI-powered insurance analytics platform that classifies vehicle damage from images, estimates repair costs, predicts claim severity, detects potential fraud, and explains model predictions using Explainable AI.

 **Live Demo:** https://vehicle-damage-risk-platform-muocmd6xrn6aqdzkgdorfs.streamlit.app/ 
 **GitHub Repository:** https://github.com/aspabhi31/vehicle-damage-risk-platform

---

##  Project Overview

Insurance companies process thousands of claims every day. Manual assessment of vehicle damage and fraud investigation is time-consuming and expensive.

This project automates the entire claim analysis workflow using machine learning and deep learning.

### Key Capabilities

-  **Vehicle Damage Classification** using :contentReference[oaicite:1]{index=1} CNN
-  **Repair Cost Estimation** using Regression
-  **Claim Severity Prediction** using Classification
-  **Fraud Risk Scoring** using :contentReference[oaicite:2]{index=2}
-  **Model Explainability** using :contentReference[oaicite:3]{index=3}
-  **Interactive Web Application** built with :contentReference[oaicite:4]{index=4}

---

##  End-to-End Pipeline

```text
Vehicle Image
     ↓
Damage Classification (CNN)
     ↓
Repair Cost Estimation
     ↓
Claim Severity Prediction
     ↓
Fraud Risk Scoring
     ↓
SHAP Explainability
     ↓
Interactive Streamlit Dashboard
