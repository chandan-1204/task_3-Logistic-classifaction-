# Logistic Regression – Binary Classification (Task 4)

This project implements a complete Logistic Regression pipeline for binary classification using the dataset `data.csv`.  
It includes preprocessing, model training, evaluation, ROC/AUC plots, precision-recall curves, threshold tuning, and saving outputs.

------------------------------------------------------------

## Project Structure

logistic_classification.py   # Main script
logreg_outputs/              # Auto-generated output folder
│
├── confusion_matrix.png
├── confusion_matrix_thresh_<value>.png
├── roc_curve.png
├── precision_recall_curve.png
├── probability_dist_by_class.png
├── threshold_metrics.csv
├── evaluation_metrics.csv
└── logreg_pipeline.joblib   # Saved trained model

------------------------------------------------------------

## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- joblib

Install all dependencies:

pip install pandas numpy matplotlib seaborn scikit-learn joblib

------------------------------------------------------------

## How the Script Works

1. Loads the dataset (data.csv).
2. Automatically detects the binary target column.
3. Normalizes target values into 0/1.
4. Splits data into train/test (80/20, stratified).
5. Preprocesses features:
   - Numeric: median imputation + StandardScaler  
   - Categorical: most-frequent imputation + OneHotEncoding
6. Builds a Scikit-Learn Pipeline combining preprocessing and LogisticRegression.
7. Trains the logistic regression model.
8. Evaluates performance using:
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - Confusion Matrix
   - ROC Curve + AUC
   - Precision-Recall Curve
9. Tunes classification thresholds (0.1 to 0.9).
10. Selects the best threshold based on F1-score.
11. Saves plots and CSV reports into `logreg_outputs/`.
12. Saves trained model as `logreg_pipeline.joblib`.

------------------------------------------------------------

## How to Run

1. Place `data.csv` in your project folder.
2. Run the script:

python logistic_classification.py

3. All outputs will be generated inside:

logreg_outputs/

------------------------------------------------------------

## Outputs Explained

- confusion_matrix.png  
  Confusion matrix at default threshold (0.5)

- confusion_matrix_thresh_<value>.png  
  Confusion matrix using the best threshold

- roc_curve.png  
  ROC curve with AUC score

- precision_recall_curve.png  
  Precision–Recall curve (important for imbalanced datasets)

- probability_dist_by_class.png  
  Histogram of predicted probabilities for both classes

- threshold_metrics.csv  
  Precision, Recall, and F1-score for thresholds 0.1–0.9

- evaluation_metrics.csv  
  Accuracy, Precision, Recall, F1-score, and ROC-AUC

- logreg_pipeline.joblib  
  Saved trained logistic regression model

------------------------------------------------------------

## What You Learn

- Binary classification
- Logistic regression fundamentals
- Sigmoid function behavior
- Data preprocessing and ML pipelines
- Evaluation metrics
- ROC curve and AUC
- Precision–recall analysis
- Threshold tuning for better classification
- Saving and loading ML models

------------------------------------------------------------

## Troubleshooting

Dataset not found:  
Ensure `data.csv` is in the same folder as the script.

NameError: '_' is not defined:  
Remove any stray underscore lines inside the script.

scikit-learn import error:  
Reinstall using:
pip install --upgrade scikit-learn

Categorical column error:  
Ensure column names are valid (no special symbols, no empty names).

------------------------------------------------------------

