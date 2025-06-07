# AIML427 Assignment 3 – Group Component

This repository contains the group work for Assignment 3 in AIML427: Big Data (2025), which involves applying Spark MLlib algorithms (Decision Tree and Logistic Regression) to the KDD Cup 1999 dataset for network anomaly detection.

## 📌 Objectives

- Use Apache Spark to train and evaluate Decision Tree and Logistic Regression models on a large-scale dataset.
- Run experiments across multiple random seeds to assess model performance (accuracy, stability, runtime).
- Demonstrate teamwork, reproducibility, and clear documentation.

## 🧪 Models

- `decision_tree_kdd.py`: Applies Spark's `DecisionTreeClassifier` to the preprocessed dataset.
- `logistic_regression_kdd.py`: Applies Spark's `LogisticRegression` model.
- Both scripts accept a random seed as command-line argument for consistent splits.

## 📊 Results

Each model was run 10 times with different seeds. Outputs include:
- Accuracy statistics (min, max, avg, std dev)
- Runtime
- Result summaries in `results/`

## 📂 Structure

- `report/`: the written group report
- `README_run.txt`: how to install, configure, and run everything
