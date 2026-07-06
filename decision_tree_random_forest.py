"""
Decision Trees & Random Forests - Heart Disease Classification Pipeline

This script implements a complete end-to-end machine learning pipeline for
predicting heart disease. It handles data loading, preprocessing (duplicate
removal, categorical encoding), model training (Decision Tree with pruning,
Random Forest), evaluation, cross-validation, feature importance analysis,
and saves all required visualizations and model artifacts.

Author: Senior Machine Learning Engineer
Date: July 2026
"""

import os
import joblib
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join("outputs", "pipeline.log"), mode="w")
    ]
)
logger = logging.getLogger(__name__)


def create_directories():
    """Ensure all required output directories exist."""
    directories = ["dataset", "notebooks", "images", "models", "outputs"]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")


def load_and_preprocess_data(filepath):
    """
    Load dataset and perform preprocessing.
    
    1. Loads dataset from CSV.
    2. Identifies and removes duplicate records to prevent data leakage.
    3. Handles missing values if any.
    4. One-hot encodes nominal categorical features.
    
    Args:
        filepath (str): Path to the dataset CSV file.
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, preprocessed_feature_names)
    """
    logger.info(f"Loading dataset from: {filepath}")
    df = pd.read_csv(filepath)
    
    # Dataset dimensions & sanity check
    logger.info(f"Initial dataset shape: {df.shape}")
    logger.info(f"Missing values count per column:\n{df.isnull().sum()}")
    
    # Check and remove duplicates
    duplicate_count = df.duplicated().sum()
    logger.info(f"Number of duplicate rows found: {duplicate_count}")
    if duplicate_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        logger.info(f"Shape after removing duplicates: {df.shape}")
    
    # Target column analysis
    target_counts = df['target'].value_counts()
    logger.info(f"Class distribution in target column:\n{target_counts}")
    
    # Define categorical features with multiple levels
    # cp: chest pain type (4 levels)
    # restecg: resting electrocardiographic results (3 levels)
    # slope: slope of peak exercise ST segment (3 levels)
    # thal: thalassemia (4 levels)
    categorical_cols = ['cp', 'restecg', 'slope', 'thal']
    
    # Convert nominal variables to string categories to ensure correct one-hot encoding
    for col in categorical_cols:
        df[col] = df[col].astype(str)
        
    # Apply One-Hot Encoding to categorical variables
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    logger.info(f"Shape after One-Hot Encoding: {df_encoded.shape}")
    
    # Separate features and target
    X = df_encoded.drop(columns=['target'])
    y = df_encoded['target']
    
    # Stratified Train-Test Split (80/20 ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    logger.info(f"Training set shape: {X_train.shape}")
    logger.info(f"Test set shape: {X_test.shape}")
    
    # Save a preview of the dataset
    save_dataset_preview(df.head(10))
    
    return X_train, X_test, y_train, y_test, X.columns.tolist()


def save_dataset_preview(df_head):
    """Generate and save a visual preview of the dataset."""
    plt.figure(figsize=(12, 4))
    plt.axis('off')
    tbl = plt.table(
        cellText=df_head.values,
        colLabels=df_head.columns,
        loc='center',
        cellLoc='center'
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.2, 1.5)
    plt.title("Dataset Preview (First 10 Rows)", fontsize=14, pad=20, weight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join("images", "dataset_preview.png"), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved dataset_preview.png")


def plot_class_distribution(y):
    """Plot distribution of classes in the target variable."""
    plt.figure(figsize=(6, 5))
    sns.set_theme(style="whitegrid")
    
    ax = sns.countplot(x=y, palette="coolwarm", hue=y, legend=False)
    total = len(y)
    for p in ax.patches:
        height = p.get_height()
        percentage = (height / total) * 100
        ax.annotate(
            f'{height}\n({percentage:.1f}%)',
            (p.get_x() + p.get_width() / 2., height),
            ha='center', va='bottom', fontsize=11, color='black', xytext=(0, 5),
            textcoords='offset points'
        )
        
    plt.title("Class Distribution (Heart Disease Target)", fontsize=14, weight='bold', pad=15)
    plt.xlabel("Heart Disease Status (0 = No Disease, 1 = Disease)", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.xticks([0, 1], ["No Disease", "Heart Disease"])
    plt.ylim(0, total * 0.7)
    plt.tight_layout()
    plt.savefig(os.path.join("images", "class_distribution.png"), dpi=300)
    plt.close()
    logger.info("Saved class_distribution.png")


def plot_correlation_heatmap(filepath):
    """Plot correlation heatmap for the continuous numerical features."""
    df = pd.read_csv(filepath)
    df = df.drop_duplicates()
    
    # Continuous features in the UCI dataset
    continuous_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    corr_matrix = df[continuous_cols].corr()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        vmin=-1,
        vmax=1,
        cbar_kws={"shrink": .8}
    )
    plt.title("Correlation Heatmap (Continuous Variables)", fontsize=14, weight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join("images", "correlation_heatmap.png"), dpi=300)
    plt.close()
    logger.info("Saved correlation_heatmap.png")


def perform_overfitting_analysis(X_train, X_test, y_train, y_test):
    """
    Train Decision Trees at various depths to analyze overfitting behavior.
    """
    depths = list(range(1, 16))
    train_accs = []
    test_accs = []
    
    for depth in depths:
        dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
        dt.fit(X_train, y_train)
        
        train_pred = dt.predict(X_train)
        test_pred = dt.predict(X_test)
        
        train_accs.append(accuracy_score(y_train, train_pred))
        test_accs.append(accuracy_score(y_test, test_pred))
        
    logger.info("Decision Tree Depth Overfitting Analysis:")
    for depth, train_acc, test_acc in zip(depths, train_accs, test_accs):
        logger.info(f"Depth {depth:2d} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
        
    return depths, train_accs, test_accs


def train_decision_tree(X_train, y_train, X_test, y_test, feature_names):
    """
    Train a Decision Tree Classifier using hyperparameter tuning to prevent overfitting.
    """
    logger.info("Tuning Decision Tree via Grid Search...")
    
    # Sweep depths, min samples split, and cost complexity pruning parameter ccp_alpha
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [3, 4, 5, 6, 7, 8],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4, 6]
    }
    
    dt_base = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(
        dt_base, param_grid, cv=5, scoring='accuracy', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    best_dt = grid_search.best_estimator_
    logger.info(f"Best Decision Tree Parameters: {grid_search.best_params_}")
    
    # Fit and evaluate
    best_dt.fit(X_train, y_train)
    
    # Save the Decision Tree model
    joblib.dump(best_dt, os.path.join("models", "decision_tree.pkl"))
    logger.info("Saved decision_tree.pkl to models/")
    
    # Save tree visualization
    plt.figure(figsize=(20, 10))
    plot_tree(
        best_dt,
        feature_names=feature_names,
        class_names=["No Disease", "Disease"],
        filled=True,
        rounded=True,
        fontsize=10
    )
    plt.title(f"Optimized Decision Tree Structure (Max Depth = {best_dt.max_depth})", fontsize=16, weight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join("images", "decision_tree.png"), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved decision_tree.png")
    
    return best_dt


def train_random_forest(X_train, y_train):
    """
    Train a Random Forest Classifier with standard tuned hyperparameters.
    """
    logger.info("Training Random Forest Classifier...")
    
    # Define hyperparameter candidates
    param_grid = {
        'n_estimators': [100, 150, 200],
        'max_depth': [4, 5, 6, 7],
        'min_samples_split': [2, 5],
        'bootstrap': [True]
    }
    
    rf_base = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        rf_base, param_grid, cv=5, scoring='accuracy', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    best_rf = grid_search.best_estimator_
    logger.info(f"Best Random Forest Parameters: {grid_search.best_params_}")
    
    # Fit
    best_rf.fit(X_train, y_train)
    
    # Save Random Forest model
    joblib.dump(best_rf, os.path.join("models", "random_forest.pkl"))
    logger.info("Saved random_forest.pkl to models/")
    
    return best_rf


def evaluate_model_performance(model, X_test, y_test, model_name):
    """
    Compute key classification metrics on the test set.
    """
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probs)
    }
    
    logger.info(f"=== {model_name} Evaluation ===")
    for k, v in metrics.items():
        logger.info(f"{k.capitalize()}: {v:.4f}")
        
    conf_matrix = confusion_matrix(y_test, preds)
    class_report = classification_report(y_test, preds)
    
    logger.info(f"Confusion Matrix:\n{conf_matrix}")
    logger.info(f"Classification Report:\n{class_report}")
    
    return metrics, conf_matrix, class_report, preds, probs


def plot_confusion_matrix(conf_matrix, model_name, filename):
    """Plot confusion matrix heatmap."""
    plt.figure(figsize=(6, 5))
    sns.set_theme(style="white")
    
    # Generate labels
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_counts = [f"{value:0.0f}" for value in conf_matrix.flatten()]
    group_percentages = [f"{value:.1%}" for value in conf_matrix.flatten() / np.sum(conf_matrix)]
    
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    
    sns.heatmap(
        conf_matrix,
        annot=labels,
        fmt="",
        cmap="Blues",
        cbar=False,
        xticklabels=["No Disease", "Disease"],
        yticklabels=["No Disease", "Disease"],
        annot_kws={"fontsize": 12, "weight": "bold"}
    )
    
    plt.title(f"Confusion Matrix - {model_name}", fontsize=14, weight='bold', pad=15)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join("images", filename), dpi=300)
    plt.close()
    logger.info(f"Saved {filename}")


def plot_model_comparison(dt_metrics, rf_metrics):
    """Plot bar chart comparing model performance metrics."""
    metrics_names = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    
    dt_vals = [dt_metrics["accuracy"], dt_metrics["precision"], dt_metrics["recall"], dt_metrics["f1"], dt_metrics["roc_auc"]]
    rf_vals = [rf_metrics["accuracy"], rf_metrics["precision"], rf_metrics["recall"], rf_metrics["f1"], rf_metrics["roc_auc"]]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    plt.figure(figsize=(9, 6))
    sns.set_theme(style="whitegrid")
    
    rects1 = plt.bar(x - width/2, dt_vals, width, label='Decision Tree', color='#8ab4f8')
    rects2 = plt.bar(x + width/2, rf_vals, width, label='Random Forest', color='#174ea6')
    
    plt.ylabel('Scores', fontsize=12)
    plt.title('Model Evaluation Comparison', fontsize=14, weight='bold', pad=15)
    plt.xticks(x, metrics_names, fontsize=11)
    plt.ylim(0, 1.1)
    plt.legend(frameon=True, facecolor='white', loc='lower right')
    
    # Annotate bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            plt.annotate(
                f'{height:.3f}',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom', fontsize=9
            )
            
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig(os.path.join("images", "model_comparison.png"), dpi=300)
    plt.close()
    logger.info("Saved model_comparison.png")


def evaluate_cross_validation(dt_model, rf_model, X_train, y_train):
    """
    Perform 5-fold cross-validation on both models and save scores distribution plot.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    dt_cv_scores = cross_val_score(dt_model, X_train, y_train, cv=cv, scoring='accuracy')
    rf_cv_scores = cross_val_score(rf_model, X_train, y_train, cv=cv, scoring='accuracy')
    
    logger.info("=== 5-Fold Cross Validation (Training Set) ===")
    logger.info(f"Decision Tree CV Scores: {dt_cv_scores}")
    logger.info(f"Decision Tree CV Mean: {dt_cv_scores.mean():.4f} (std: {dt_cv_scores.std():.4f})")
    logger.info(f"Random Forest CV Scores: {rf_cv_scores}")
    logger.info(f"Random Forest CV Mean: {rf_cv_scores.mean():.4f} (std: {rf_cv_scores.std():.4f})")
    
    # Plot CV Scores comparison
    plt.figure(figsize=(7, 5))
    sns.set_theme(style="whitegrid")
    
    cv_df = pd.DataFrame({
        "Decision Tree": dt_cv_scores,
        "Random Forest": rf_cv_scores
    })
    
    sns.boxplot(data=cv_df, palette="coolwarm", width=0.4)
    sns.stripplot(data=cv_df, color="black", size=6, jitter=0.1)
    
    plt.title("5-Fold Cross Validation Accuracy Scores", fontsize=14, weight='bold', pad=15)
    plt.ylabel("Accuracy", fontsize=12)
    plt.ylim(min(dt_cv_scores.min(), rf_cv_scores.min()) - 0.05, 1.02)
    plt.tight_layout()
    plt.savefig(os.path.join("images", "cross_validation_scores.png"), dpi=300)
    plt.close()
    logger.info("Saved cross_validation_scores.png")
    
    return dt_cv_scores, rf_cv_scores


def plot_feature_importance(dt_model, rf_model, feature_names):
    """Plot feature importances side-by-side."""
    dt_importances = dt_model.feature_importances_
    rf_importances = rf_model.feature_importances_
    
    feat_imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Decision Tree': dt_importances,
        'Random Forest': rf_importances
    })
    
    # Sort by Random Forest importance
    feat_imp_df = feat_imp_df.sort_values(by='Random Forest', ascending=True)
    
    # Plot horizontal comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharey=True)
    sns.set_theme(style="whitegrid")
    
    # Decision Tree Importance
    dt_sorted = feat_imp_df.sort_values(by='Decision Tree', ascending=True)
    sns.barplot(
        x='Decision Tree', y='Feature', data=dt_sorted,
        ax=axes[0], color='#8ab4f8'
    )
    axes[0].set_title('Decision Tree Feature Importance', fontsize=12, weight='bold')
    axes[0].set_xlabel('Importance Score', fontsize=11)
    axes[0].set_ylabel('Features', fontsize=11)
    
    # Random Forest Importance
    sns.barplot(
        x='Random Forest', y='Feature', data=feat_imp_df,
        ax=axes[1], color='#174ea6'
    )
    axes[1].set_title('Random Forest Feature Importance', fontsize=12, weight='bold')
    axes[1].set_xlabel('Importance Score', fontsize=11)
    axes[1].set_ylabel('')
    
    plt.suptitle('Feature Importance Comparison', fontsize=16, weight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join("images", "feature_importance.png"), dpi=300)
    plt.close()
    logger.info("Saved feature_importance.png")


def save_evaluation_report(dt_metrics, rf_metrics, dt_cv, rf_cv, dt_report, rf_report, dt_params, rf_params):
    """Save comprehensive evaluation summary to a text file."""
    report_path = os.path.join("outputs", "evaluation_metrics.txt")
    with open(report_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("DECISION TREE & RANDOM FOREST PIPELINE EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("1. DATA PREPROCESSING SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write("Original UCI Dataset: 1,025 resampled records.\n")
        f.write("Deduplicated Dataset: 302 unique records (dropped 723 duplicate rows to avoid data leakage).\n")
        f.write("Train/Test Split: 80% train (241 rows), 20% test (61 rows), stratified.\n\n")
        
        f.write("2. MODEL PARAMETERS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Decision Tree Best Parameters: {dt_params}\n")
        f.write(f"Random Forest Best Parameters: {rf_params}\n\n")
        
        f.write("3. TEST SET PERFORMANCE METRICS\n")
        f.write("-" * 40 + "\n")
        f.write("Decision Tree Classifier:\n")
        f.write(f"  - Accuracy:  {dt_metrics['accuracy']:.4f}\n")
        f.write(f"  - Precision: {dt_metrics['precision']:.4f}\n")
        f.write(f"  - Recall:    {dt_metrics['recall']:.4f}\n")
        f.write(f"  - F1 Score:  {dt_metrics['f1']:.4f}\n")
        f.write(f"  - ROC-AUC:   {dt_metrics['roc_auc']:.4f}\n\n")
        
        f.write("Random Forest Classifier:\n")
        f.write(f"  - Accuracy:  {rf_metrics['accuracy']:.4f}\n")
        f.write(f"  - Precision: {rf_metrics['precision']:.4f}\n")
        f.write(f"  - Recall:    {rf_metrics['recall']:.4f}\n")
        f.write(f"  - F1 Score:  {rf_metrics['f1']:.4f}\n")
        f.write(f"  - ROC-AUC:   {rf_metrics['roc_auc']:.4f}\n\n")
        
        f.write("4. 5-FOLD CROSS VALIDATION ACCURACY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Decision Tree Mean CV: {dt_cv.mean():.4f} (std: {dt_cv.std():.4f})\n")
        f.write(f"Random Forest Mean CV: {rf_cv.mean():.4f} (std: {rf_cv.std():.4f})\n\n")
        
        f.write("5. CLASSIFICATION REPORTS\n")
        f.write("-" * 40 + "\n")
        f.write("Decision Tree Classification Report:\n")
        f.write(dt_report)
        f.write("\nRandom Forest Classification Report:\n")
        f.write(rf_report)
        
    logger.info(f"Saved evaluation metrics report to {report_path}")


def main():
    """Execute the end-to-end pipeline."""
    create_directories()
    
    # Step 1: Preprocessing
    dataset_path = os.path.join("dataset", "heart_disease.csv")
    if not os.path.exists(dataset_path):
        # Fail-safe check: copy heart.csv to dataset directory if missing
        if os.path.exists("heart.csv"):
            import shutil
            shutil.copy("heart.csv", dataset_path)
            logger.info("Copied heart.csv to dataset/heart_disease.csv")
        else:
            raise FileNotFoundError("Could not find heart.csv to initialize the pipeline.")
            
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data(dataset_path)
    
    # Step 2: Basic distributions and heatmaps
    plot_class_distribution(y_train)
    plot_correlation_heatmap(dataset_path)
    
    # Step 3: Overfitting analysis (Decision Tree depth plot in notebook)
    perform_overfitting_analysis(X_train, X_test, y_train, y_test)
    
    # Step 4: Model Training
    dt_model = train_decision_tree(X_train, y_train, X_test, y_test, feature_names)
    rf_model = train_random_forest(X_train, y_train)
    
    # Step 5: Evaluation
    dt_metrics, dt_conf, dt_report, dt_preds, dt_probs = evaluate_model_performance(dt_model, X_test, y_test, "Decision Tree")
    rf_metrics, rf_conf, rf_report, rf_preds, rf_probs = evaluate_model_performance(rf_model, X_test, y_test, "Random Forest")
    
    # Step 6: Save Confusion Matrices & Model Comparison
    plot_confusion_matrix(dt_conf, "Decision Tree", "confusion_matrix_dt.png")
    plot_confusion_matrix(rf_conf, "Random Forest", "confusion_matrix_rf.png")
    plot_model_comparison(dt_metrics, rf_metrics)
    
    # Step 7: Cross Validation
    dt_cv, rf_cv = evaluate_cross_validation(dt_model, rf_model, X_train, y_train)
    
    # Step 8: Feature Importance
    plot_feature_importance(dt_model, rf_model, feature_names)
    
    # Step 9: Save predictions as CSV files
    predictions_dt_df = pd.DataFrame({
        'true_label': y_test,
        'predicted_label': dt_preds,
        'predicted_probability': dt_probs
    })
    predictions_dt_df.to_csv(os.path.join("outputs", "predictions_dt.csv"), index=False)
    
    predictions_rf_df = pd.DataFrame({
        'true_label': y_test,
        'predicted_label': rf_preds,
        'predicted_probability': rf_probs
    })
    predictions_rf_df.to_csv(os.path.join("outputs", "predictions_rf.csv"), index=False)
    
    # Step 10: Save complete metrics report
    save_evaluation_report(
        dt_metrics, rf_metrics, dt_cv, rf_cv, dt_report, rf_report,
        dt_model.get_params(), rf_model.get_params()
    )
    
    logger.info("Pipeline executed successfully. All models, evaluations, and figures saved.")


if __name__ == '__main__':
    main()
