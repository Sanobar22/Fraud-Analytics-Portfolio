# Global Fraud Analytics and Applied AI Portfolio
Architected by Sanobar Shaikh

## Executive Summary
This portfolio presents a unified, multi-domain AI ecosystem engineered to detect and mitigate financial crime. Developed by Sanobar Shaikh, these projects bridge the gap between high-dimensional financial data and actionable risk intelligence.

## Why This Portfolio
The financial sector is currently facing a convergence of traditional fraud and AI-augmented threats. This portfolio is designed to demonstrate a proactive defense strategy. Instead of isolated models, it showcases a holistic view of the fraud lifecycle—from transactional anomalies and synthetic identity creation to deepfake-augmented social engineering.

## What Sets Me Apart
1. Multi-Domain Integration: While most analysts focus on a single fraud type, my work spans Transactional, Behavioral, and Media Forensics.
2. Production-Ready Thinking: The transition from Jupyter Notebooks to a functional Streamlit Dashboard demonstrates my ability to deploy models into real-world business environments.
3. Hybrid Methodology: I combine Supervised Learning (XGBoost, Random Forest) for known patterns with Unsupervised Learning (Isolation Forest) to detect emerging, "zero-day" fraud tactics.

## Interactive Dashboard
The entire portfolio is accessible through a professional web-based interface that allows for real-time risk assessment and model interaction.

### Running the Dashboard
1. Clone the repository:
   git clone https://github.com/Sanobar22/Fraud-Analytics-Portfolio.git
2. Navigate to the project directory:
   cd Fraud-Analytics-Portfolio
3. Install the required dependencies:
   pip install -r requirements.txt
4. Execute the application:
   streamlit run app.py

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
Folder: /transaction-fraud-detection
Algorithm: Random Forest with SMOTE optimization.
Technical Note: Leverages PCA-transformed features to maintain data privacy while achieving high precision in risk flagging.

### 2. Deepfake Media Forensics
Objective: Detecting AI-generated synthetic faces in identity verification (eKYC) processes.
Folder: /Deepfake Detection with Neural Network
Algorithm: Convolutional Neural Network (CNN).
Technical Note: Focuses on detecting spatial texture anomalies characteristic of GAN-generated imagery.

### 3. Identity Fraud Detection
Objective: Identifying synthetic identities and account takeover attempts.
Folder: /Identity-Fraud-Detection
Algorithm: XGBoost Gradient Boosting.
Technical Note: Analyzes behavioral metadata including device fingerprints and geolocation to build a risk profile for every session.

### 4. Check Kiting Detection
Objective: Identifying illegal "float" schemes in check-based transactions.
Folder: /Check Kiting Detection using Unsupervised learning
Algorithm: Isolation Forest (Anomaly Detection).
Technical Note: Detects outliers in transaction velocity and balance shifts without the need for historical labels.

### 5. Compliance and AML Monitoring
Objective: Automating regulatory screening and anti-money laundering monitoring.
Folder: /KYC-AML Compliance system
Focus: Risk-weighting algorithms and threshold-based monitoring for high-risk jurisdictions.

---

## Contact
Sanobar Shaikh
LinkedIn: https://www.linkedin.com/in/sanobar-shaikh/
GitHub: https://github.com/Sanobar22
