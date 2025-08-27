# MBTI Personality Prediction: A Data-Centric Analysis

This project presents a comprehensive data analysis and modeling pipeline for predicting Myers-Briggs (MBTI) personality types from text. Moving beyond a simple model application, it focuses on **feature engineering** and **robust evaluation** to derive insights from a noisy, real-world dataset.

## Key Highlights

*   **Data Preprocessing & Cleaning:** Implemented a full text cleaning pipeline including lowercasing, stopword removal, and lemmatization to prepare user-generated forum posts for analysis.
*   **Advanced Feature Engineering:** Engineered two novel feature sets:
    *   **Trait Vectors:** A graph-based feature constructed using PageRank propagation over trait co-occurrence graphs to model psychological associations.
    *   **Stylometric Features:** Quantified writing style through metrics like sentence length, punctuation usage, and lexical diversity.
*   **Comparative Model Analysis:** Systematically evaluated the performance of **seven machine learning models** (including Logistic Regression, Random Forest, XGBoost, LightGBM) across 98 configurations to understand which approaches work best for different data representations.
*   **Handling Data Imbalance:** Addressed significant class imbalance in the dataset using SMOTE, providing a clear comparison of its effects on each model and feature set.

This work demonstrates a full-cycle data analysis project, from data cleaning and exploratory analysis to feature creation, model testing, and result interpretation.
