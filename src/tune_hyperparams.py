# tune_hyperparams.py
# Phase 2: hyperparameter optimization with Optuna
# Optimizes the top 4 models from Phase 1 (LogReg, CatBoost, LightGBM, XGBoost)

import time
import json
import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocess import preprocess
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
optuna.logging.set_verbosity(optuna.logging.WARNING) # Silence Optuna logs


# Settings
N_TRIALS = 100   # number of Optuna trials per model
N_FOLDS = 5      # cross-validation folds
SEED = 42
cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)


# Loading
print('Loading data...')
X_train_tree, _, _, y_train, _, _, _ = preprocess(scale_numeric=False)
X_train_log, _, _, y_train_log, _, _, _ = preprocess(scale_numeric=True)


# Objective functions for each model
# Each function defines the search space and returns the AUC to maximize.

def objective_logreg(trial):
    # Logistic regression: tune regularization
    params = {
        'C': trial.suggest_float('C', 0.001, 10, log=True),
        'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
        'solver': 'liblinear',  # supports both l1 and l2
        'max_iter': 2000,
        'class_weight': 'balanced',
        'random_state': SEED,
    }
    model = LogisticRegression(**params)
    scores = cross_val_score(model, X_train_log, y_train_log,
                             cv=cv, scoring='roc_auc', n_jobs=-1)
    return scores.mean()


def objective_xgboost(trial):
    # XGBoost: tune main hyperparameters
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 600),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 1e-3, 10, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10, log=True),
        'scale_pos_weight': (y_train == 0).sum() / (y_train == 1).sum(),
        'random_state': SEED,
        'eval_metric': 'logloss',
        'n_jobs': -1,
    }
    model = XGBClassifier(**params)
    scores = cross_val_score(model, X_train_tree, y_train,
                             cv=cv, scoring='roc_auc', n_jobs=-1)
    return scores.mean()


def objective_lightgbm(trial):
    # LightGBM: tune main hyperparameters
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 600),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 15, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10, log=True),
        'class_weight': 'balanced',
        'random_state': SEED,
        'verbose': -1,
        'n_jobs': -1,
    }
    model = LGBMClassifier(**params)
    scores = cross_val_score(model, X_train_tree, y_train,
                             cv=cv, scoring='roc_auc', n_jobs=-1)
    return scores.mean()


def objective_catboost(trial):
    # CatBoost: tune main hyperparameters
    params = {
        'iterations': trial.suggest_int('iterations', 100, 600),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'depth': trial.suggest_int('depth', 3, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10, log=True),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'random_strength': trial.suggest_float('random_strength', 0.1, 10, log=True),
        'auto_class_weights': 'Balanced',
        'random_seed': SEED,
        'verbose': False,
    }
    model = CatBoostClassifier(**params)
    scores = cross_val_score(model, X_train_tree, y_train,
                             cv=cv, scoring='roc_auc', n_jobs=-1)
    return scores.mean()


# Running the studies
studies = {}

models_to_tune = {
    'LogisticRegression': objective_logreg,
    'XGBoost': objective_xgboost,
    'LightGBM': objective_lightgbm,
    'CatBoost': objective_catboost,
}

for name, objective in models_to_tune.items():
    print(f'\nTuning {name}... ({N_TRIALS} trials)')
    start = time.time()

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    elapsed = time.time() - start
    studies[name] = study

    print(f'  Best AUC: {study.best_value:.4f}')
    print(f'  Best params: {study.best_params}')
    print(f'  Time: {elapsed:.1f}s')


# Summary
print('TUNING RESULTS SUMMARY')

summary = []
for name, study in studies.items():
    summary.append({
        'model': name,
        'best_auc': study.best_value,
        'n_trials': len(study.trials),
    })

summary_df = pd.DataFrame(summary).sort_values('best_auc', ascending=False)
print(summary_df.to_string(index=False))

best_model_name = summary_df.iloc[0]['model']
print(f'\nBest model after tuning: {best_model_name} '
      f'(AUC = {summary_df.iloc[0]["best_auc"]:.4f})')


# Saving the best hyperparameters
best_params = {name: study.best_params for name, study in studies.items()}
best_params['_best_model'] = best_model_name
best_params['_best_auc'] = float(summary_df.iloc[0]['best_auc'])

with open('../outputs/best_hyperparams.json', 'w') as f:
    json.dump(best_params, f, indent=2)

print('\nSaved best hyperparameters to ../outputs/best_hyperparams.json')


# Optimization history plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for ax, (name, study) in zip(axes, studies.items()):
    aucs = [t.value for t in study.trials if t.value is not None]
    best_so_far = np.maximum.accumulate(aucs)

    ax.plot(aucs, alpha=0.4, label='Each trial')
    ax.plot(best_so_far, color='red', linewidth=2, label='Best so far')
    ax.set_title(f'{name} : Optuna optimization', fontweight='bold')
    ax.set_xlabel('Trial')
    ax.set_ylabel('AUC-ROC')
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('../outputs/optuna_optimization_history.png', dpi=200, bbox_inches='tight')
plt.show()