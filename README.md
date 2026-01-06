Project About: This Project is Part of "Zerve AI Hackethon Compitition". Project is about selecting real Business Problam and solve using create a ML Model, Tunned, getting matricx and Deployed by creating API for results checks.

ML Model : "Production-Ready Customer Churn Prediction System"

Description

This project implements a production-ready customer churn prediction system built entirely on the Zerve platform.

The system ingests telecom customer data, performs feature preprocessing and scaling, trains a Gradient Boosting classification model, and evaluates performance using business-relevant metrics including ROC-AUC, precision, recall, F1-score, and confusion matrices.

A hyperparameter optimization step is included to improve model effectiveness using cross-validated tuning, after which the best-performing model is selected for deployment.

The final model is deployed as a Zerve task that generates real-time churn predictions, probability scores, risk classification (LOW / MEDIUM / HIGH), confidence estimates, and actionable retention recommendations.

A lightweight web interface demonstrates real-world usage by allowing users to input customer profiles and visualize predictions and model performance. This system is designed to reflect a realistic, production-grade decision-support workflow for customer retention teams.

1. Problem Statement

Customer churn is a critical business challenge in subscription-based industries such as telecom, where retaining existing customers is significantly more cost-effective than acquiring new ones.
The objective of this project is to build a deployed analytical system that predicts customer churn risk and enables proactive retention actions.

2. Solution Overview

I built an end-to-end customer churn prediction system entirely on the Zerve platform.
The solution consists of data preprocessing, machine learning model training, validation, hyperparameter optimization, and deployment as a production-ready task.

The system accepts customer attributes as input and returns:

Churn prediction (will churn / will stay)

Churn probability

Risk level classification (LOW / MEDIUM / HIGH)

Confidence score

Actionable business recommendations

3. Dataset

The model was trained on a telecom customer churn dataset containing 7,043 customers, with a 33.4% churn rate, representing a realistic, moderately imbalanced business scenario.

4. Model & Validation Approach

A Gradient Boosting classifier was selected due to its strong performance on structured tabular data and suitability for production use.

Baseline Evaluation

The baseline model was evaluated using an 80/20 train-validation split with the following metrics:

Accuracy

Precision

Recall

F1-score

ROC-AUC

Confusion matrix

Hyperparameter Optimization

To improve effectiveness, a hyperparameter tuning step was added using cross-validated search with ROC-AUC as the optimization objective.
The optimized model demonstrated improved discrimination capability compared to the baseline and was selected for deployment.

This structured baseline → tuning → selection workflow reflects real-world ML best practices.

5. Model Performance (Optimized Model)

The final deployed model achieved:

ROC-AUC: ~0.76+, indicating strong ability to distinguish churners from non-churners

Balanced precision and recall suitable for cost-effective retention strategies

Stable performance across customer segments

ROC-AUC was prioritized because churn prediction is a ranking and decision-support problem rather than a pure accuracy task.

6. Deployment & Production Readiness

The optimized model is deployed as a Zerve Task, enabling on-demand execution within the platform.
The deployed system includes:

Input validation

Feature preprocessing consistent with training

Batch and single-record prediction support

Deterministic and reproducible execution

A lightweight web interface demonstrates how the deployed model can be consumed by end users and business teams.

7. Business Impact

This system enables organizations to:

Identify high-risk customers early

Focus retention efforts on the most valuable segments

Reduce churn-related revenue loss

Make data-driven retention decisions using probability-based risk scores

The solution is designed to be directly extensible to real production environments.
