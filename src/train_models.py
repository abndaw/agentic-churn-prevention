# train_models.py
# Phase 1: quick benchmark of 5 models with default hyperparameters
# 5-fold stratified cross-validation on the training set

import time
import numpy as np
import pandas as pd
import seaborn as sns
from preprocess import preprocess
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate


# 1. Loading the preprocessed data
print('Loading preprocessed data...')
X_train, X_val, X_test, y_train, y_val, y_test, feature_names = preprocess(scale_numeric=False)


# 2. The models to test
# class_weight='balanced' will handle the moderate class imbalance
# All models use random_state=42 for reproducibility
models = {
    'LogisticRegression': LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    ),

    'RandomForest': RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ),

    'XGBoost': XGBClassifier(
        n_estimators=100,
        # XGBoost : scale_pos_weight instead of class_weight
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        random_state=42,
        eval_metric='logloss',
        n_jobs=-1
    ),

    'LightGBM': LGBMClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42,
        verbose=-1,
        n_jobs=-1
    ),

    'CatBoost': CatBoostClassifier(
        iterations=100,
        auto_class_weights='Balanced',
        random_seed=42,
        verbose=False
    ),
}


# 3. Cross-validation setup
# 5-fold stratified to preserve the churn ratio in each fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Metrics we will track
scoring = {
    'auc': 'roc_auc',
    'f1': 'f1',
    'precision': 'precision',
    'recall': 'recall',
    'pr_auc': 'average_precision',
}


# 4. Running the benchmark
results = []

for name, model in models.items():
    print(f'\nTraining {name}...')

    start = time.time()

    # For LogisticRegression : scaled data
    if name == 'LogisticRegression':
        X_tr_scaled, _, _, y_tr_scaled, _, _, _ = preprocess(scale_numeric=True)
        scores = cross_validate(model, X_tr_scaled, y_tr_scaled,
                                cv=cv, scoring=scoring, n_jobs=-1)
    else:
        scores = cross_validate(model, X_train, y_train,
                                cv=cv, scoring=scoring, n_jobs=-1)

    elapsed = time.time() - start

    # Average across the 5 folds
    results.append({
        'model': name,
        'auc': scores['test_auc'].mean(),
        'auc_std': scores['test_auc'].std(),
        'f1': scores['test_f1'].mean(),
        'precision': scores['test_precision'].mean(),
        'recall': scores['test_recall'].mean(),
        'pr_auc': scores['test_pr_auc'].mean(),
        'time_sec': elapsed,
    })

    print(f'  AUC = {scores["test_auc"].mean():.4f} '
          f'(+/- {scores["test_auc"].std():.4f}) | '
          f'time = {elapsed:.1f}s')


# 5. Saving and displaying the results
results_df = pd.DataFrame(results).sort_values('auc', ascending=False)

# Rounded version for display
display_df = results_df.copy()
for col in ['auc', 'auc_std', 'f1', 'precision', 'recall', 'pr_auc']:
    display_df[col] = display_df[col].round(4)
display_df['time_sec'] = display_df['time_sec'].round(1)

print('\n' + '=' * 80)
print('BENCHMARK RESULTS (5-fold stratified CV on training set)')
print('=' * 80)
print(display_df.to_string(index=False))

results_df.to_csv('../outputs/benchmark_results.csv', index=False)

# 6. Comparison plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: AUC with error bars
sns.barplot(data=results_df, x='model', y='auc',
            ax=axes[0], palette='viridis')
axes[0].errorbar(x=range(len(results_df)),
                 y=results_df['auc'],
                 yerr=results_df['auc_std'],
                 fmt='none', c='black', capsize=5)
axes[0].set_title('AUC-ROC by model (5-fold CV, mean ± std)', fontweight='bold')
axes[0].set_ylim(0.7, 0.9)
axes[0].set_ylabel('AUC-ROC')
axes[0].set_xlabel('')
for i, v in enumerate(results_df['auc']):
    axes[0].text(i, v + 0.005, f'{v:.3f}', ha='center', fontweight='bold')

# Plot 2: F1 / Precision / Recall comparison
metrics_long = results_df.melt(id_vars='model',
                                value_vars=['f1', 'precision', 'recall'],
                                var_name='metric', value_name='score')
sns.barplot(data=metrics_long, x='model', y='score',
            hue='metric', ax=axes[1], palette='Set2')
axes[1].set_title('F1 / Precision / Recall by model', fontweight='bold')
axes[1].set_ylim(0, 1)
axes[1].set_xlabel('')

plt.tight_layout()
plt.savefig('../outputs/benchmark_comparison.png', dpi=200, bbox_inches='tight')
plt.show()