# Heart Disease Prediction

This repository compares a Decision Tree Classifier and a Random Forest Classifier to predict the presence of heart disease from clinical patient data. The project handles preprocessing, hyperparameter optimization via grid search, cross-validation, and feature importance analysis.

---

## Project Overview

* **Objective**: Predict whether a patient has heart disease (1 = diseased, 0 = healthy).
* **Data Leakage Resolution**: Identified and resolved a data duplication issue (1,025 raw rows reduced to 302 unique patient profiles), ensuring validation integrity.
* **Results Summary**:
  * Both models achieved **80.33% accuracy** on the test split (61 samples).
  * The Random Forest showed stronger generalization, with a **ROC-AUC of 88.42%** (vs. 83.77% for the Decision Tree) and a **5-fold cross-validation accuracy of 81.72%** (vs. 74.73% for the Decision Tree).
  * Key predictors identified include thalassemia (`thal`) and chest pain type (`cp`).

---

## Repository Structure

```
.
├── dataset/
│   └── heart.csv                 # Patient records
│
├── notebooks/
│   └── decision_tree_random_forest.ipynb # Analysis and rendering notebook
│
├── images/                               # Saved plots and figures
│   ├── dataset_preview.png
│   ├── class_distribution.png
│   ├── correlation_heatmap.png
│   ├── decision_tree.png
│   ├── confusion_matrix_dt.png
│   ├── confusion_matrix_rf.png
│   ├── feature_importance.png
│   ├── model_comparison.png
│   └── cross_validation_scores.png
│
├── models/                               # Serialized models
│   ├── decision_tree.pkl
│   └── random_forest.pkl
│
├── outputs/                              # Pipeline output predictions and log
│   ├── predictions_dt.csv
│   ├── predictions_rf.csv
│   ├── evaluation_metrics.txt
│   └── pipeline.log
│
├── decision_tree_random_forest.py       # Python pipeline execution script
├── requirements.txt                     # Dependencies
└── README.md                            # Project overview
```

---

## Preprocessing & Methodology

1. **Deduplication**: The raw dataset contains duplicate entries representing identical patient records. Splitting the data prior to deduplication causes data leakage, as identical records end up in both splits. We drop duplicates first to produce 302 unique profiles (164 heart disease cases, 138 healthy controls).
2. **Encoding**: Nominal features (`cp`, `restecg`, `slope`, `thal`) are one-hot encoded to avoid imposing ordinal relationships.
3. **Scaling**: No feature scaling is applied since tree-based classifiers are scale-invariant, preserving feature interpretability.

---

## Model Performance

The hyperparameters were tuned using 5-fold stratified cross-validation on the training set:
* **Decision Tree**: `{'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 6, 'min_samples_split': 2}`
* **Random Forest**: `{'bootstrap': True, 'max_depth': 7, 'min_samples_split': 2, 'n_estimators': 150}`

### Test Set Metrics Comparison

| Metric | Decision Tree | Random Forest | Delta |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 80.33% | 80.33% | 0.00% |
| **Precision** | 81.82% | 81.82% | 0.00% |
| **Recall** | 81.82% | 81.82% | 0.00% |
| **F1-Score** | 81.82% | 81.82% | 0.00% |
| **ROC-AUC** | **83.77%** | **88.42%** | **+4.65% (RF Superior)** |
| **Mean CV Accuracy** | **74.73%** | **81.72%** | **+6.99% (RF Superior)** |

*Note*: Test accuracies are identical due to the small size of the holdout test set (61 cases). The cross-validation and ROC-AUC scores provide a more reliable measure of the Random Forest's superior generalizability.

---

## Saved Visualizations

The pipeline generates and saves the following plots to the `images/` directory:

1. **EDA Previews**:
   * `dataset_preview.png`: First 10 rows of the dataset.
   * `class_distribution.png`: Count of target classes.
   * `correlation_heatmap.png`: Correlation matrix of continuous features.
2. **Model Structures & Outputs**:
   * `decision_tree.png`: Plot of the optimized Decision Tree structure.
   * `confusion_matrix_dt.png` / `confusion_matrix_rf.png`: Confusion matrices for both classifiers.
   * `model_comparison.png`: Comparison of performance metrics.
   * `cross_validation_scores.png`: Boxplot of accuracy scores across 5 folds.
   * `feature_importance.png`: Feature importances compared side-by-side.

---

## Usage

### Setup
Install dependencies:
```bash
pip install -r requirements.txt
```

### Run the Pipeline
Execute the Python script to run the preprocessing, train models, save output predictions, metrics, and plots:
```bash
python decision_tree_random_forest.py
```

### Run the Notebook
To view the analysis step-by-step with math explanations:
```bash
jupyter notebook notebooks/decision_tree_random_forest.ipynb
```

---

## License
Licensed under the [MIT License](LICENSE).
