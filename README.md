# 📊 Auto Insurance & Telematics Analytics Portfolio

Welcome to my data science portfolio. This repository features two end-to-end machine learning projects focused on the automotive and insurance industries. Together, they demonstrate the ability to extract actionable business intelligence from both unlabelled telematics sensors and highly imbalanced insurance datasets.

The goal of this repository is to showcase end-to-end analytical thinking, robust statistical methodologies, and machine learning models built specifically to drive business value.

---

### 🛠️ Core Tech Stack
**Languages & Frameworks:** Python (>=3.14)
**Libraries:** `scikit-learn`, `pandas`, `numpy`, `statsmodels`, `shap`, `plotly`, `matplotlib`, `seaborn`

---

### 🗺️ Project Architecture Overview
Below is the interactive flowchart outlining the data engineering and modeling processes used across the projects.

*Click the image below to view and interact with the full flowchart!*
[![Project Flow Chart](charts/flow%20chart%20prev.png)](charts/interactive%20flow%20chart.html)

---

## 🛡️ Project 1: Predictive Modeling for Car Insurance Claims
> **File:** [`insurance claim.ipynb`](insurance%20claim.ipynb) | **Status:** Completed ✅ | **Performance:** 0.65 ROC-AUC & 64% Recall 🏆

### 📖 Business Context
Predicting car insurance claims is notoriously difficult due to extreme class imbalance (only ~6% of policies result in a claim in this dataset). Traditional accuracy metrics are misleading in this context, masking the business cost of missed claims. This project builds a robust machine learning pipeline to identify high-risk drivers and physical vehicle attributes, translating complex non-linear patterns into actionable risk-management strategies.

### 🎯 Key Objectives
1. **Eradicate Multicollinearity:** Automate the removal of structural multicollinearity (e.g., overlapping car physical dimensions and engine specs) using distributed Variance Inflation Factor (VIF) calculations.
2. **Statistical Inference:** Utilize Stepwise Logistic Regression to distill 80+ features into the core mathematical drivers of insurance risk.
3. **Predictive Machine Learning:** Execute a hyperparameter-tuned "Model Bake-Off" between Linear, Bagging, and Boosting algorithms.
4. **Business Threshold Optimization:** Recalibrate the model's decision threshold to maximize the capture of actual claims (Recall) without overwhelming the business with false alarms.

### ⚙️ The Workflow
1. **Data Engineering & Feature Selection:** Cleaned string representations of torque/power, mapped binaries, and dynamically bucketed sparse geographic categories (< 1% frequency). Dropped 56 highly correlated features using a strict VIF threshold of 5.0.
2. **Statistical Modeling:** Deployed a custom Stepwise Logistic Regression algorithm (p-value < 0.05) to isolate 9 statistically significant features. *Insight: Older cars dramatically reduce claim risk, while premium feature packages (`is_brake_assist_Yes`) and policy tenure increase it.*
3. **Model Bake-Off:** Conducted a `GridSearchCV` on Logistic Regression, Random Forest, and HistGradientBoosting using `class_weight='balanced'`. Random Forest outperformed other models with the highest ROC-AUC score, proving the presence of non-linear risk factors.
4. **Threshold Tuning:** Calculated Youden's J Statistic to optimize the decision boundary for the imbalanced reality of the data.

### 📈 Results & Business Impact
By shifting the focus from standard accuracy to optimized Recall, the final Random Forest model delivered significant business value:
* **High Recall (64%):** The model successfully captured 64% of the entire risk pool (746 out of 1161 actual claims) from unseen test data.
* **Precision Lift:** While precision naturally sits at 10%, the baseline standard for a claim is only 6%. The model successfully isolated a specific sub-population of drivers whose risk of crashing is nearly double the average, justifying targeted premium adjustments.

---

## 🚗 Project 2: Driver Behavior Profiling (Unsupervised Telematics)
> **File:** [`labeling drivers.ipynb`](labeling%20drivers.ipynb) | **Status:** Completed ✅ | **Validation Score:** 1.00 ARI 🏆

### 📖 Business Context
In real-world Usage-Based Insurance (UBI), driver behavior data is abundant but rarely pre-labeled. This project builds a Machine Learning pipeline to automatically categorize drivers into distinct behavioral profiles (**Safe**, **Aggressive**, **Distracted**) using raw telematics data. 

**The "Real-World" Twist:** Although the original dataset contained labeled data (ground truth), this project intentionally treats the problem as Unsupervised Learning. It demonstrates how to use **Dimensionality Reduction (PCA)** and **Clustering (K-Means)** to mathematically discover these behaviors from scratch.

### 🎯 Key Objectives
1. **Simulate Real-World Conditions:** Train a model without access to the target variable (`driver_label`).
2. **Dimensionality Reduction:** Compress noisy sensor data into meaningful "Behavioral Dimensions."
3. **Automatic Profiling:** Group drivers based on their PCA coordinates.
4. **Validation:** Compare the unsupervised clusters against the hidden ground truth to measure the model's ability to "recover" true labels.

### ⚙️ The Workflow
1. **Correlation Analysis:** Identified an **"Aggression Box"** where Speed, Acceleration, and Braking were highly multicollinear.
2. **Dimensionality Reduction (PCA):** Reduced the dataset to **3 components**, explaining **~80%** of the variance:
    * **PC1 (Aggression):** High `throttle`, `brake_pressure`, `accel_x`
    * **PC2 (Distraction):** High `phone_usage`, `reaction_time`, `lane_deviation`
    * **PC3 (Geometry):** Purely `steering_angle`
3. **Unsupervised Clustering (K-Means):** Applied K-Means (k=3) to segment drivers into Safe, Distracted, and Aggressive clusters.

### 📊 Interactive Cluster Visualization
*Click the image below to explore the interactive 3D scatter plot of the behavioral clusters!*
[![Driver Behavior Clusters](charts/scatter%20prev.png)](charts/interactive%20scatter.html)

### 📈 Results & Validation
Using the withheld labels to validate the unsupervised learning:
* **Adjusted Rand Score (ARI):** 1.00
* **Homogeneity Score:** 1.00

**Conclusion:** The model achieved a perfect score, proving that driver behaviors are mathematically distinct and linearly separable in the PCA space. We can effectively auto-label new telematics data with 100% accuracy without needing human supervision, providing a scalable solution for telematics-based risk assessment.
