# Global Fraud Analytics and Applied AI Portfolio
By Sanobar Shaikh

## Executive Summary
This portfolio presents a unified, multi-domain AI ecosystem engineered to detect and mitigate financial crime. Developed by Sanobar Shaikh, these projects bridge the gap between high-dimensional financial data and actionable risk intelligence.

## Why This Portfolio
The financial sector is currently facing a convergence of traditional fraud and AI-augmented threats. This portfolio is designed to demonstrate a proactive defense strategy. Instead of isolated models, it showcases a holistic view of the fraud lifecycle—from transactional anomalies and synthetic identity creation to deepfake-augmented social engineering.

## What Sets Me Apart
1. Multi-Domain Integration: While most analysts focus on a single fraud type, my work spans Transactional, Behavioral, and Media Forensics.
2. Production-Ready Thinking: The transition from Jupyter Notebooks to a functional Streamlit Dashboard demonstrates my ability to deploy models into real-world business environments.
3. Hybrid Methodology: I combine Supervised Learning (XGBoost, Random Forest) for known patterns with Unsupervised Learning (Isolation Forest) to detect emerging, "zero-day" fraud tactics.

## Model Performance and Application Matrix

| Fraud Domain | AI Methodology | Primary Technical Focus | Business Application |
| :--- | :--- | :--- | :--- |
| Transaction Fraud | Random Forest + SMOTE | Class Imbalance Management | Real-time Card Authorization |
| Deepfake Forensics | Convolutional Neural Networks | Spatial Texture Analysis | eKYC Identity Verification |
| Identity Fraud | XGBoost Gradient Boosting | Behavioral Metadata Analysis | Account Opening Protection |
| Check Kiting | Isolation Forest | Unsupervised Anomaly Detection | Liquidity Risk Management |
| AML Compliance | Risk-Weighting Heuristics | Threshold Monitoring | Regulatory Reporting (SAR) |

## Interactive Dashboard
The entire portfolio is accessible through a professional web-based interface that allows for real-time risk assessment and model interaction.

### Running the Dashboard

Follow these steps to initialize and launch the application:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Sanobar22/Fraud-Analytics-Portfolio.git
   ```

2. **Navigate to the project directory:**
   ```bash
   cd Fraud-Analytics-Portfolio
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Execute the application:**
   ```bash
   streamlit run app.py
   ```

---

## Core Skills and Tools
1. Programming and Frameworks: Python (Pandas, NumPy, Scikit-Learn), TensorFlow, Keras, XGBoost.
2. Analytical Methodologies: SMOTE for imbalanced data, PCA for dimensionality reduction, Anomaly Detection, Media Forensics.
3. Deployment and Visualization: Streamlit, Plotly, Joblib for model persistence.
4. Evaluation Metrics: ROC-AUC, Precision-Recall curves, F1-Score optimization.

---

## Project Domains

### 1. Transaction Fraud Detection
Objective: Real-time classification of fraudulent credit card transactions.  
Algorithm: Random Forest with SMOTE optimization.  
Technical Note: Leverages PCA-transformed features to maintain data privacy while achieving high precision in risk flagging.  
[View Project](./transaction-fraud-detection)

### 2. Deepfake Media Forensics
Objective: Detecting AI-generated synthetic faces in identity verification (eKYC) processes.  
Algorithm: Convolutional Neural Network (CNN).  
Technical Note: Focuses on detecting spatial texture anomalies characteristic of GAN-generated imagery.  
[View Project](./Deepfake%20Detection%20with%20Neural%20Network)

### 3. Identity Fraud Detection
Objective: Identifying synthetic identities and account takeover attempts.  
Algorithm: XGBoost Gradient Boosting.  
Technical Note: Analyzes behavioral metadata including device fingerprints and geolocation to build a risk profile for every session.  
[View Project](./Identity-Fraud-Detection)

### 4. Check Kiting Detection
Objective: Identifying illegal "float" schemes in check-based transactions.  
Algorithm: Isolation Forest (Anomaly Detection).  
Technical Note: Detects outliers in transaction velocity and balance shifts without the need for historical labels.  
[View Project](./Check%20Kiting%20Detection%20using%20Unsupervised%20learning)

### 5. Compliance and AML Monitoring
Objective: Automating regulatory screening and anti-money laundering monitoring.  
Focus: Risk-weighting algorithms and threshold-based monitoring for high-risk jurisdictions.  
[View Project](./KYC-AML%20Compliance%20system)

---

## Contact
Sanobar Shaikh  
LinkedIn: https://www.linkedin.com/in/sanobar-shaikh/  
GitHub: https://github.com/Sanobar22
