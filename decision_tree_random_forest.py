"""
Heart disease classification using Decision Trees and Random Forests.
Handles preprocessing, grid search optimization, validation, and plotting.
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
    roc_auc_score, confusion_matrix, classification_report
)

# Set up simple logging configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)

os.makedirs("outputs", exist_ok=True)
fh = logging.FileHandler(os.path.join("outputs", "pipeline.log"), mode="w")
fh.setFormatter(formatter)
logger.addHandler(fh)


def load_and_preprocess(filepath):
    """Load heart.csv, drop duplicates, and one-hot encode categoricals."""
    logger.info(f"Loading dataset from: {filepath}")
    df = pd.read_csv(filepath)
    
    # Drop duplicate records to prevent data leakage between train/test splits
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        logger.info(f"Found {duplicates} duplicate rows. Dropping duplicates...")
        df = df.drop_duplicates().reset_index(drop=True)
    
    logger.info(f"Dataset shape: {df.shape}")
    
    # Nominal categorical columns to encode
    cat_cols = ['cp', 'restecg', 'slope', 'thal']
    for col in cat_cols:
        df[col] = df[col].astype(str)
        
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    X = df_encoded.drop(columns=['target'])
    y = df_encoded['target']
    
    # Stratified split to maintain class ratio in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    save_dataset_preview(df.head(10))
    
    return X_train, X_test, y_train, y_test, X.columns.tolist()


def save_dataset_preview(df_head):
    """Save a PNG table of the first few rows of the dataframe."""
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
    logger.info("Saved dataset preview image.")


def plot_class_distribution(y):
    """Save countplot of target class labels."""
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
    logger.info("Saved target class distribution plot.")


def plot_correlation_heatmap(filepath):
    """Generate correlation heatmap for numeric features."""
    df = pd.read_csv(filepath).drop_duplicates()
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
    plt.title("Correlation Heatmap", fontsize=14, weight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join("images", "correlation_heatmap.png"), dpi=300)
    plt.close()
    logger.info("Saved correlation heatmap.")


def perform_overfitting_analysis(X_train, X_test, y_train, y_test):
    """Evaluate decision tree accuracy at varying depths to find overfitting threshold."""
    depths = list(range(1, 16))
    train_accs, test_accs = [], []
    
    for depth in depths:
        dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
        dt.fit(X_train, y_train)
        train_accs.append(accuracy_score(y_train, dt.predict(X_train)))
        test_accs.append(accuracy_score(y_test, dt.predict(X_test)))
        
    logger.info("Overfitting Analysis (Decision Tree Depth):")
    for d, tr, te in zip(depths, train_accs, test_accs):
        logger.info(f"  Depth {d:2d} -> Train Acc: {tr:.4f} | Test Acc: {te:.4f}")
        
    return depths, train_accs, test_accs


def train_decision_tree(X_train, y_train, feature_names):
    """Tune and train a Decision Tree Classifier using Grid Search."""
    logger.info("Tuning Decision Tree hyperparameters...")
    
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [3, 4, 5, 6, 7, 8],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4, 6]
    }
    
    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid, cv=5, scoring='accuracy', n_jobs=-1
    )
    grid.fit(X_train, y_train)
    
    best_dt = grid.best_estimator_
    logger.info(f"Best Decision Tree params: {grid.best_params_}")
    
    joblib.dump(best_dt, os.path.join("models", "decision_tree.pkl"))
    
    plt.figure(figsize=(20, 10))
    plot_tree(
        best_dt,
        feature_names=feature_names,
        class_names=["No Disease", "Disease"],
        filled=True,
        rounded=True,
        fontsize=10
    )
    plt.title(f"Decision Tree Structure (Depth = {best_dt.max_depth})", fontsize=16, weight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join("images", "decision_tree.png"), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved Decision Tree structure plot.")
    
    return best_dt


def train_random_forest(X_train, y_train):
    """Tune and train a Random Forest Classifier using Grid Search."""
    logger.info("Tuning Random Forest hyperparameters...")
    
    param_grid = {
        'n_estimators': [100, 150, 200],
        'max_depth': [4, 5, 6, 7],
        'min_samples_split': [2, 5],
        'bootstrap': [True]
    }
    
    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid, cv=5, scoring='accuracy', n_jobs=-1
    )
    grid.fit(X_train, y_train)
    
    best_rf = grid.best_estimator_
    logger.info(f"Best Random Forest params: {grid.best_params_}")
    
    joblib.dump(best_rf, os.path.join("models", "random_forest.pkl"))
    return best_rf


def evaluate_model(model, X_test, y_test, name):
    """Calculate and log classification performance metrics."""
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probs)
    }
    
    logger.info(f"--- {name} Performance ---")
    for metric_name, val in metrics.items():
        logger.info(f"{metric_name.replace('_', ' ').capitalize()}: {val:.4f}")
        
    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds)
    
    logger.info(f"Confusion Matrix:\n{cm}")
    logger.info(f"Classification Report:\n{report}")
    
    return metrics, cm, report, preds, probs


def plot_confusion_matrix(cm, model_name, filename):
    """Plot labeled confusion matrix heatmap."""
    plt.figure(figsize=(6, 5))
    sns.set_theme(style="white")
    
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_counts = [f"{v:0.0f}" for v in cm.flatten()]
    group_pcts = [f"{v:.1%}" for v in cm.flatten() / cm.sum()]
    
    labels = [f"{n}\n{c}\n{p}" for n, c, p in zip(group_names, group_counts, group_pcts)]
    labels = np.asarray(labels).reshape(2, 2)
    
    sns.heatmap(
        cm,
        annot=labels,
        fmt="",
        cmap="Blues",
        cbar=False,
        xticklabels=["No Disease", "Disease"],
        yticklabels=["No Disease", "Disease"],
        annot_kws={"fontsize": 12, "weight": "bold"}
    )
    
    plt.title(f"Confusion Matrix ({model_name})", fontsize=14, weight='bold', pad=15)
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("Actual", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join("images", filename), dpi=300)
    plt.close()
    logger.info(f"Saved confusion matrix: {filename}")


def plot_model_comparison(dt_metrics, rf_metrics):
    """Bar chart comparing decision tree and random forest metrics."""
    metrics_names = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    
    dt_vals = [dt_metrics["accuracy"], dt_metrics["precision"], dt_metrics["recall"], dt_metrics["f1"], dt_metrics["roc_auc"]]
    rf_vals = [rf_metrics["accuracy"], rf_metrics["precision"], rf_metrics["recall"], rf_metrics["f1"], rf_metrics["roc_auc"]]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    plt.figure(figsize=(9, 6))
    sns.set_theme(style="whitegrid")
    
    rects1 = plt.bar(x - width/2, dt_vals, width, label='Decision Tree', color='#8ab4f8')
    rects2 = plt.bar(x + width/2, rf_vals, width, label='Random Forest', color='#174ea6')
    
    plt.ylabel('Score')
    plt.title('Model Evaluation Comparison', fontsize=14, weight='bold', pad=15)
    plt.xticks(x, metrics_names, fontsize=11)
    plt.ylim(0, 1.1)
    plt.legend(loc='lower right')
    
    def annotate_bars(rects):
        for rect in rects:
            h = rect.get_height()
            plt.annotate(
                f'{h:.3f}',
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=9
            )
            
    annotate_bars(rects1)
    annotate_bars(rects2)
    
    plt.tight_layout()
    plt.savefig(os.path.join("images", "model_comparison.png"), dpi=300)
    plt.close()
    logger.info("Saved model comparison plot.")


def evaluate_cross_validation(dt, rf, X_train, y_train):
    """Run stratified 5-fold cross-validation on both models."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    dt_scores = cross_val_score(dt, X_train, y_train, cv=cv, scoring='accuracy')
    rf_scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring='accuracy')
    
    logger.info("--- 5-Fold Cross Validation ---")
    logger.info(f"Decision Tree CV: {dt_scores.mean():.4f} (std: {dt_scores.std():.4f})")
    logger.info(f"Random Forest CV: {rf_scores.mean():.4f} (std: {rf_scores.std():.4f})")
    
    plt.figure(figsize=(7, 5))
    sns.set_theme(style="whitegrid")
    
    cv_df = pd.DataFrame({"Decision Tree": dt_scores, "Random Forest": rf_scores})
    sns.boxplot(data=cv_df, palette="coolwarm", width=0.4)
    sns.stripplot(data=cv_df, color="black", size=6, jitter=0.1)
    
    plt.title("5-Fold Cross-Validation Accuracy", fontsize=14, weight='bold', pad=15)
    plt.ylabel("Accuracy")
    plt.ylim(min(dt_scores.min(), rf_scores.min()) - 0.05, 1.02)
    plt.tight_layout()
    plt.savefig(os.path.join("images", "cross_validation_scores.png"), dpi=300)
    plt.close()
    logger.info("Saved cross-validation score comparison plot.")
    
    return dt_scores, rf_scores


def plot_feature_importance(dt, rf, feature_names):
    """Compare feature importances of Decision Tree and Random Forest side-by-side."""
    feat_imp = pd.DataFrame({
        'Feature': feature_names,
        'Decision Tree': dt.feature_importances_,
        'Random Forest': rf.feature_importances_
    })
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharey=True)
    sns.set_theme(style="whitegrid")
    
    dt_sorted = feat_imp.sort_values(by='Decision Tree')
    sns.barplot(x='Decision Tree', y='Feature', data=dt_sorted, ax=axes[0], color='#8ab4f8')
    axes[0].set_title('Decision Tree Feature Importance', fontsize=12, weight='bold')
    axes[0].set_xlabel('Score')
    axes[0].set_ylabel('Features')
    
    rf_sorted = feat_imp.sort_values(by='Random Forest')
    sns.barplot(x='Random Forest', y='Feature', data=rf_sorted, ax=axes[1], color='#174ea6')
    axes[1].set_title('Random Forest Feature Importance', fontsize=12, weight='bold')
    axes[1].set_xlabel('Score')
    axes[1].set_ylabel('')
    
    plt.suptitle('Feature Importance Comparison', fontsize=16, weight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join("images", "feature_importance.png"), dpi=300)
    plt.close()
    logger.info("Saved feature importance comparison plot.")


def save_evaluation_report(dt_metrics, rf_metrics, dt_cv, rf_cv, dt_report, rf_report, dt_params, rf_params):
    """Write performance metrics and comparison to a text file."""
    report_path = os.path.join("outputs", "evaluation_metrics.txt")
    with open(report_path, "w") as f:
        f.write("Heart Disease Prediction Model Evaluation Report\n")
        f.write("================================================\n\n")
        
        f.write("Data Preprocessing Summary\n")
        f.write("--------------------------\n")
        f.write("Original dataset: 1,025 resampled records.\n")
        f.write("Deduplicated dataset: 302 unique patient profiles (dropped 723 duplicates to prevent leakage).\n")
        f.write("Split: 80% training (241 samples), 20% test (61 samples), stratified.\n\n")
        
        f.write("Model Configurations\n")
        f.write("--------------------\n")
        f.write(f"Decision Tree Best Parameters: {dt_params}\n")
        f.write(f"Random Forest Best Parameters: {rf_params}\n\n")
        
        f.write("Test Set Metrics\n")
        f.write("----------------\n")
        f.write("Decision Tree Classifier:\n")
        f.write(f"  Accuracy:  {dt_metrics['accuracy']:.4f}\n")
        f.write(f"  Precision: {dt_metrics['precision']:.4f}\n")
        f.write(f"  Recall:    {dt_metrics['recall']:.4f}\n")
        f.write(f"  F1 Score:  {dt_metrics['f1']:.4f}\n")
        f.write(f"  ROC-AUC:   {dt_metrics['roc_auc']:.4f}\n\n")
        
        f.write("Random Forest Classifier:\n")
        f.write(f"  Accuracy:  {rf_metrics['accuracy']:.4f}\n")
        f.write(f"  Precision: {rf_metrics['precision']:.4f}\n")
        f.write(f"  Recall:    {rf_metrics['recall']:.4f}\n")
        f.write(f"  F1 Score:  {rf_metrics['f1']:.4f}\n")
        f.write(f"  ROC-AUC:   {rf_metrics['roc_auc']:.4f}\n\n")
        
        f.write("5-Fold Cross Validation Accuracy\n")
        f.write("--------------------------------\n")
        f.write(f"Decision Tree Mean CV: {dt_cv.mean():.4f} (std: {dt_cv.std():.4f})\n")
        f.write(f"Random Forest Mean CV: {rf_cv.mean():.4f} (std: {rf_cv.std():.4f})\n\n")
        
        f.write("Classification Reports\n")
        f.write("----------------------\n")
        f.write("Decision Tree:\n")
        f.write(dt_report)
        f.write("\nRandom Forest:\n")
        f.write(rf_report)
        
    logger.info(f"Saved evaluation metrics report to {report_path}")


def main():
    # Make sure output directories exist
    os.makedirs("images", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    dataset_path = os.path.join("dataset", "heart.csv")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Could not find dataset file at {dataset_path}")
            
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess(dataset_path)
    
    # Exploratory Data Analysis plots
    plot_class_distribution(y_train)
    plot_correlation_heatmap(dataset_path)
    
    # Evaluate overfitting trends with varying tree depths
    perform_overfitting_analysis(X_train, X_test, y_train, y_test)
    
    # Tune and train the classifiers
    dt_model = train_decision_tree(X_train, y_train, feature_names)
    rf_model = train_random_forest(X_train, y_train)
    
    # Evaluate models on test set
    dt_metrics, dt_conf, dt_report, dt_preds, dt_probs = evaluate_model(dt_model, X_test, y_test, "Decision Tree")
    rf_metrics, rf_conf, rf_report, rf_preds, rf_probs = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    
    # Generate evaluation plots
    plot_confusion_matrix(dt_conf, "Decision Tree", "confusion_matrix_dt.png")
    plot_confusion_matrix(rf_conf, "Random Forest", "confusion_matrix_rf.png")
    plot_model_comparison(dt_metrics, rf_metrics)
    
    # Model validation and feature importance
    dt_cv, rf_cv = evaluate_cross_validation(dt_model, rf_model, X_train, y_train)
    plot_feature_importance(dt_model, rf_model, feature_names)
    
    # Save predictions
    pd.DataFrame({
        'true_label': y_test,
        'predicted_label': dt_preds,
        'predicted_probability': dt_probs
    }).to_csv(os.path.join("outputs", "predictions_dt.csv"), index=False)
    
    pd.DataFrame({
        'true_label': y_test,
        'predicted_label': rf_preds,
        'predicted_probability': rf_probs
    }).to_csv(os.path.join("outputs", "predictions_rf.csv"), index=False)
    
    # Write the summary report
    save_evaluation_report(
        dt_metrics, rf_metrics, dt_cv, rf_cv, dt_report, rf_report,
        dt_model.get_params(), rf_model.get_params()
    )
    
    logger.info("Pipeline completed successfully. All artifacts saved.")


if __name__ == '__main__':
    main()
