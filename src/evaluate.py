# evaluate.py
# Phase 3: Model selection on validation set

import shap
import json
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from preprocess import preprocess
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    roc_curve, confusion_matrix, classification_report, 
    average_precision_score
)

# Loading tuned hyperparameters
with open('../outputs/best_hyperparams.json', 'r') as f:
    best_params = json.load(f)

print('Loaded hyperparameters from tuning phase:')
print(f'  Best CV model was: {best_params["_best_model"]}')
print(f'  Best CV AUC: {best_params["_best_auc"]:.4f}')

# Loading data
X_train, X_val, X_test, y_train, y_val, y_test, feature_names = preprocess(scale_numeric=False)
X_train_log, X_val_log, X_test_log, _, _, _, _ = preprocess(scale_numeric=True)

print(f'\nData splits:')
print(f'  Train: {X_train.shape}')
print(f'  Val:   {X_val.shape}')
print(f'  Test:  {X_test.shape}')

print('STEP 1: TRAINING ALL MODELS AND EVALUATE ON VALIDATION SET')

validation_results = []
trained_models = {}

# Logistic Regression
print('\n[1/4] Training LogisticRegression')
lr = LogisticRegression(
    C=best_params['LogisticRegression']['C'],
    penalty=best_params['LogisticRegression']['penalty'],
    solver='liblinear',
    max_iter=2000,
    class_weight='balanced',
    random_state=42
)
lr.fit(X_train_log, y_train)
lr_val_proba = lr.predict_proba(X_val_log)[:, 1]
lr_val_pred = lr.predict(X_val_log)
trained_models['LogisticRegression'] = (lr, True)  # True = needs scaled data

validation_results.append({
    'Model': 'LogisticRegression',
    'Val_AUC': roc_auc_score(y_val, lr_val_proba),
    'Val_F1': f1_score(y_val, lr_val_pred),
    'Val_Precision': precision_score(y_val, lr_val_pred),
    'Val_Recall': recall_score(y_val, lr_val_pred),
})
print(f'  Validation AUC: {validation_results[-1]["Val_AUC"]:.4f}')

# XGBoost
print('\n[2/4] Training XGBoost')
xgb = XGBClassifier(
    n_estimators=best_params['XGBoost']['n_estimators'],
    learning_rate=best_params['XGBoost']['learning_rate'],
    max_depth=best_params['XGBoost']['max_depth'],
    min_child_weight=best_params['XGBoost']['min_child_weight'],
    subsample=best_params['XGBoost']['subsample'],
    colsample_bytree=best_params['XGBoost']['colsample_bytree'],
    gamma=best_params['XGBoost']['gamma'],
    reg_alpha=best_params['XGBoost']['reg_alpha'],
    reg_lambda=best_params['XGBoost']['reg_lambda'],
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
    random_state=42,
    eval_metric='logloss',
    n_jobs=-1
)
xgb.fit(X_train, y_train)
xgb_val_proba = xgb.predict_proba(X_val)[:, 1]
xgb_val_pred = xgb.predict(X_val)
trained_models['XGBoost'] = (xgb, False)

validation_results.append({
    'Model': 'XGBoost',
    'Val_AUC': roc_auc_score(y_val, xgb_val_proba),
    'Val_F1': f1_score(y_val, xgb_val_pred),
    'Val_Precision': precision_score(y_val, xgb_val_pred),
    'Val_Recall': recall_score(y_val, xgb_val_pred),
})
print(f'  Validation AUC: {validation_results[-1]["Val_AUC"]:.4f}')

# LightGBM
print('\n[3/4] Training LightGBM')
lgbm = LGBMClassifier(
    n_estimators=best_params['LightGBM']['n_estimators'],
    learning_rate=best_params['LightGBM']['learning_rate'],
    num_leaves=best_params['LightGBM']['num_leaves'],
    max_depth=best_params['LightGBM']['max_depth'],
    min_child_samples=best_params['LightGBM']['min_child_samples'],
    subsample=best_params['LightGBM']['subsample'],
    colsample_bytree=best_params['LightGBM']['colsample_bytree'],
    reg_alpha=best_params['LightGBM']['reg_alpha'],
    reg_lambda=best_params['LightGBM']['reg_lambda'],
    class_weight='balanced',
    random_state=42,
    verbose=-1,
    n_jobs=-1
)
lgbm.fit(X_train, y_train)
lgbm_val_proba = lgbm.predict_proba(X_val)[:, 1]
lgbm_val_pred = lgbm.predict(X_val)
trained_models['LightGBM'] = (lgbm, False)

validation_results.append({
    'Model': 'LightGBM',
    'Val_AUC': roc_auc_score(y_val, lgbm_val_proba),
    'Val_F1': f1_score(y_val, lgbm_val_pred),
    'Val_Precision': precision_score(y_val, lgbm_val_pred),
    'Val_Recall': recall_score(y_val, lgbm_val_pred),
})
print(f'  Validation AUC: {validation_results[-1]["Val_AUC"]:.4f}')

# CatBoost
print('\n[4/4] Training CatBoost')
cb = CatBoostClassifier(
    iterations=best_params['CatBoost']['iterations'],
    learning_rate=best_params['CatBoost']['learning_rate'],
    depth=best_params['CatBoost']['depth'],
    l2_leaf_reg=best_params['CatBoost']['l2_leaf_reg'],
    border_count=best_params['CatBoost']['border_count'],
    random_strength=best_params['CatBoost']['random_strength'],
    auto_class_weights='Balanced',
    random_seed=42,
    verbose=False
)
cb.fit(X_train, y_train)
cb_val_proba = cb.predict_proba(X_val)[:, 1]
cb_val_pred = cb.predict(X_val)
trained_models['CatBoost'] = (cb, False)

validation_results.append({
    'Model': 'CatBoost',
    'Val_AUC': roc_auc_score(y_val, cb_val_proba),
    'Val_F1': f1_score(y_val, cb_val_pred),
    'Val_Precision': precision_score(y_val, cb_val_pred),
    'Val_Recall': recall_score(y_val, cb_val_pred),
})
print(f'  Validation AUC: {validation_results[-1]["Val_AUC"]:.4f}')

# Display validation results
val_df = pd.DataFrame(validation_results).sort_values('Val_AUC', ascending=False)

print('VALIDATION SET RESULTS (sorted by AUC)')
print(val_df.to_string(index=False))

# Save validation results
val_df.to_csv('../outputs/validation_metrics.csv', index=False)

# SELECT WINNER 
winner_name = val_df.iloc[0]['Model']
winner_val_auc = val_df.iloc[0]['Val_AUC']

print(f'WINNER: {winner_name}')
print(f'   Validation AUC: {winner_val_auc:.4f}')

# Save winner info
winner_info = {
    'winner_model': winner_name,
    'validation_auc': float(winner_val_auc),
    'selection_criterion': 'val_auc_roc',
    'cv_best_from_tuning': best_params['_best_model'],
    'cv_auc_from_tuning': best_params['_best_auc']
}
with open('../outputs/winner_selection.json', 'w') as f:
    json.dump(winner_info, f, indent=2)

print('STEP 2: VALIDATION SET SUMMARY')

# Get winner model
model, needs_scaling = trained_models[winner_name]

# Validation metrics summary
val_metrics = val_df[val_df['Model'] == winner_name].iloc[0].to_dict()

print(f'\n{winner_name} - Validation Performance:')
for k, v in val_metrics.items():
    if k != 'Model':
        print(f'  {k:20s} {v:.4f}')

# Classification report on validation
X_val_final = X_val_log if needs_scaling else X_val
y_val_pred = model.predict(X_val_final)

report = classification_report(y_val, y_val_pred, 
                              target_names=['Stayed', 'Churned'])
print('\nClassification Report (Validation Set):')
print(report)

with open('../outputs/classification_report_validation.txt', 'w') as f:
    f.write(f'Winning Model: {winner_name}\n')
    f.write(f'Validation AUC: {winner_val_auc:.4f}\n\n')
    f.write(report)

# ROC Curves 
plt.figure(figsize=(10, 7))

for model_name, (model_obj, needs_scale) in trained_models.items():
    X_val_m = X_val_log if needs_scale else X_val
    
    proba = model_obj.predict_proba(X_val_m)[:, 1]
    fpr, tpr, _ = roc_curve(y_val, proba)
    auc = roc_auc_score(y_val, proba)
    
    # Highlight winner
    if model_name == winner_name:
        plt.plot(fpr, tpr, linewidth=3, label=f'{model_name} (AUC={auc:.4f}) ★', 
                color='#e74c3c')
    else:
        plt.plot(fpr, tpr, linewidth=1.5, linestyle='--', alpha=0.7,
                label=f'{model_name} (AUC={auc:.4f})')

plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title(f'ROC Curves on Validation Set\nWinner: {winner_name}', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('../outputs/roc_curves_validation.png', dpi=300, bbox_inches='tight')
plt.close()
print('ROC curves saved')

# Confusion Matrix (on validation) 
cm = confusion_matrix(y_val, y_val_pred)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=['Stayed', 'Churned'],
            yticklabels=['Stayed', 'Churned'],
            annot_kws={'size': 14})
plt.title(f'Confusion Matrix: {winner_name}\n(Validation Set)', 
         fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('../outputs/confusion_matrix_validation.png', dpi=300, bbox_inches='tight')
plt.close()
print('Confusion matrix saved')

# SHAP Analysis (on validation)
sample_size = min(500, len(X_val_final))
X_sample = X_val_final.sample(n=sample_size, random_state=42)

if winner_name in ['XGBoost', 'LightGBM', 'CatBoost']:
    print('  Using TreeExplainer...')
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

elif winner_name == 'LogisticRegression':
    print('  Using LinearExplainer...')
    explainer = shap.LinearExplainer(model, X_train_log)
    shap_values = explainer.shap_values(X_sample)

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_sample, 
                 feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig('../outputs/shap_summary_validation.png', dpi=300, bbox_inches='tight')
plt.close()
print('SHAP summary saved')

# Saving Winner Model
with open('../models/best_model.pkl', 'wb') as f:
    pickle.dump(model, f)

metadata = {
    'model_name': winner_name,
    'feature_names': feature_names,
    'validation_auc': float(winner_val_auc),
    'needs_scaling': needs_scaling,
    'trained_on': pd.Timestamp.now().isoformat()
}

with open('../models/best_model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f'   Winner: {winner_name}')
print(f'   Validation AUC: {winner_val_auc:.4f}')
print(f'   Model saved: ../models/best_model.pkl')
print(f'   Metadata saved: ../models/best_model_metadata.json')
print(f'   Outputs saved: ../outputs/')
