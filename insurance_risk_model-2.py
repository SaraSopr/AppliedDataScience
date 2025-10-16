# # Insurance Risk Modeling (Teacher–Student Framework)
# **Goal:** Train a high-capacity **teacher** model on **all dataset features** (RandomForest or XGBoost),
# then distill it into a **compact student** model that needs **only 4 inputs** at inference time:
# `sex`, `weight`, `SMK_stat_type_cd`, `drinker_degree`.
#
# ⚠️ **Important:** The dataset has **no true actuarial target**. We create a transparent **proxy risk label** from biomedical markers.
# Replace it with real insurance outcomes (claims/premium band) for production.

# %% [markdown]
# ## 1. Setup & Imports

import os, json, warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, accuracy_score, log_loss
)
from sklearn.ensemble import RandomForestClassifier
import joblib

# Optional: XGBoost (teacher); will fallback to RandomForest if not available
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
ARTIFACT_DIR = Path('artifacts'); ARTIFACT_DIR.mkdir(exist_ok=True)

# %% [markdown]
# ## 2. Load Data & Basic Inspection

FILE = 'smoking_driking_dataset_Ver01.csv'
df = pd.read_csv(FILE)
print(df.shape)
df.head()

# %% [markdown]
# ## 3. EDA (quick)

print("\nColumns:\n", df.columns.tolist())
print("\nMissing values (top 15):\n", df.isna().sum().sort_values(ascending=False).head(15))

# Numeric summary
summary = df.describe(include='all').T
summary.head(20)

# Correlation heatmap (normalized)
numeric_df = df.select_dtypes(include=[np.number])
if len(numeric_df.columns) > 1:
    scaler = MinMaxScaler()
    norm = pd.DataFrame(scaler.fit_transform(numeric_df), columns=numeric_df.columns)
    corr = norm.corr()
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr, cmap='RdBu_r', center=0, annot=True, fmt='.2f', annot_kws={'size':7}, linewidths=.3)
    plt.title('Normalized Feature Correlation Heatmap')
    plt.tight_layout(); plt.show()

# %% [markdown]
# ## 4. Feature Engineering & Proxy Target
# - `drinker_degree` from DRK_YN (0/1)
# - BMI from height/weight
# - Transparent proxy risk score from biomedical features (HDL protective)
# - Convert to 3 bands via quantiles (Low/Medium/High)

# Deployment-friendly flags
df['drinker_degree'] = (df['DRK_YN'] == 'Y').astype(int)

# BMI
if {'height','weight'}.issubset(df.columns):
    df['BMI'] = df['weight'] / ((df['height']/100.0)**2)
else:
    df['BMI'] = np.nan

# Candidate biomedical features (use only those available)
cand_num = [
    'age','height','weight','waistline','SBP','DBP','tot_chole','HDL_chole','LDL_chole',
    'triglyceride','gamma_GTP','hemoglobin','serum_creatinine','SGOT_AST','SGOT_ALT','BMI'
]
biomarkers = [c for c in cand_num if c in df.columns]

# Build proxy score (z-scored sum; HDL is protective)
score = np.zeros(len(df))
for col in biomarkers:
    x = pd.to_numeric(df[col], errors='coerce')
    mu, sd = np.nanmean(x), np.nanstd(x)
    z = (x - mu)/(sd if sd and sd>0 else 1.0)
    score += (-1.0 if col=='HDL_chole' else 1.0) * np.nan_to_num(z)

# Light lifestyle priors
score += 0.5 * df['SMK_stat_type_cd'].replace({1:0, 2:0.5, 3:1}).fillna(0)
score += 0.25 * df['drinker_degree']

df['risk_proxy_score'] = score
q_low, q_high = np.nanpercentile(score, [33.3, 66.6])

def to_band(s):
    if s <= q_low: return 'Low'
    if s <= q_high: return 'Medium'
    return 'High'

df['risk_band'] = df['risk_proxy_score'].apply(to_band)

sns.countplot(x='risk_band', data=df, order=['Low','Medium','High'])
plt.title('Proxy Risk Bands Distribution')
plt.show()

# %% [markdown]
# ## 5. Teacher Model — Full Feature Training (XGBoost preferred, else RandomForest)
# **Inputs:**
# - Categorical: `sex`, `SMK_stat_type_cd`, `drinker_degree`
# - Numerical: all biomedical and anthropometric variables present

categorical_features = [c for c in ['sex','SMK_stat_type_cd','drinker_degree'] if c in df.columns]
numeric_features = [c for c in cand_num if c in df.columns]

X_full = df[categorical_features + numeric_features].copy()
y = df['risk_band'].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X_full, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# Preprocessing: impute + encode/scale
preprocess_full = ColumnTransformer([
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
    ]), categorical_features),
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), numeric_features)
])

# Class weights via inverse-frequency (for XGB sample_weight)
class_counts = y_train.value_counts()
class_weight = {cls: len(y_train)/ (len(class_counts)*cnt) for cls, cnt in class_counts.items()}
train_sample_weight = y_train.map(class_weight).values

# Teacher model selection
if XGB_AVAILABLE:
    teacher_clf = XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        num_class=3,
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
else:
    teacher_clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

teacher_pipe = Pipeline([
    ('preprocess', preprocess_full),
    ('clf', teacher_clf)
])

# Fit teacher
if XGB_AVAILABLE:
    teacher_pipe.fit(X_train, y_train, clf__sample_weight=train_sample_weight)
else:
    teacher_pipe.fit(X_train, y_train)

# Evaluation (teacher)
y_pred_t = teacher_pipe.predict(X_test)
y_proba_t = teacher_pipe.predict_proba(X_test)

report_t = pd.DataFrame(classification_report(y_test, y_pred_t, output_dict=True)).T
labels = sorted(y.unique())
cm_t = confusion_matrix(y_test, y_pred_t, labels=labels)

plt.figure(figsize=(6,5))
sns.heatmap(cm_t, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Teacher Confusion Matrix')
plt.xlabel('Predicted'); plt.ylabel('True'); plt.tight_layout(); plt.show()

auc_macro_t = roc_auc_score(pd.get_dummies(y_test).reindex(columns=labels, fill_value=0), y_proba_t, multi_class='ovr')
print('Teacher Macro ROC-AUC:', round(auc_macro_t, 3))
report_t

# Save teacher
teacher_name = 'xgb' if XGB_AVAILABLE else 'rf'
joblib.dump(teacher_pipe, ARTIFACT_DIR / f'risk_model_teacher_{teacher_name}.joblib')
with open(ARTIFACT_DIR / f'model_card_teacher_{teacher_name}.json', 'w') as f:
    json.dump({
        'model': f'Teacher ({"XGBoost" if XGB_AVAILABLE else "RandomForest"})',
        'features_used': categorical_features + numeric_features,
        'n_features': len(categorical_features) + len(numeric_features),
        'random_state': RANDOM_STATE,
        'notes': ['Proxy risk label; replace with actuarial target for production.']
    }, f, indent=2)
print('✅ Teacher saved to artifacts.')

# %% [markdown]
# ## 6. Student Model — 4‑Feature Distillation
# **Goal:** Learn a compact model that uses only: `sex`, `weight`, `SMK_stat_type_cd`, `drinker_degree`.
# We distill knowledge from the teacher by training the student on teacher predictions (hard labels) and
# weighting samples by teacher confidence (max class probability).

student_features = [c for c in ['sex','weight','SMK_stat_type_cd','drinker_degree'] if c in df.columns]
X_student = df[student_features].copy()

# Teacher targets (on entire dataset) for distillation
y_proba_teacher_all = teacher_pipe.predict_proba(X_full)
y_label_teacher_all = np.array(teacher_pipe.classes_)[y_proba_teacher_all.argmax(axis=1)]
y_conf_teacher_all  = y_proba_teacher_all.max(axis=1)

# Align split with original y (proxy ground truth used for stratification)
Xs_train, Xs_test, yt_train, yt_test, yh_train, yh_test, w_train, w_test = train_test_split(
    X_student, y, y_label_teacher_all, y_conf_teacher_all,
    test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# Student preprocessing
student_preprocess = ColumnTransformer([
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
    ]), [c for c in student_features if c in ['sex','SMK_stat_type_cd','drinker_degree']]),
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), [c for c in student_features if c in ['weight']])
])

# Student model: choose a light XGBoost if available, else LogisticRegression
if XGB_AVAILABLE:
    student_clf = XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        num_class=3,
        n_estimators=300,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
else:
    from sklearn.linear_model import LogisticRegression
    student_clf = LogisticRegression(multi_class='multinomial', class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE)

student_pipe = Pipeline([
    ('preprocess', student_preprocess),
    ('clf', student_clf)
])

# Fit student on teacher hard labels, weighted by teacher confidence
if XGB_AVAILABLE:
    student_pipe.fit(Xs_train, yh_train, clf__sample_weight=w_train)
else:
    student_pipe.fit(Xs_train, yh_train)

# Agreement with teacher
yh_pred_test = student_pipe.predict(Xs_test)
teacher_agreement = accuracy_score(yh_test, yh_pred_test)
print('Teacher–Student agreement (test):', round(teacher_agreement, 3))

# Performance vs ground-truth proxy (for reference)
ys_pred_test = student_pipe.predict(Xs_test)
print('\nStudent vs proxy ground-truth (test):')
print(classification_report(yt_test, ys_pred_test))

# Save student
student_name = 'xgb' if XGB_AVAILABLE else 'logreg'
joblib.dump(student_pipe, ARTIFACT_DIR / f'risk_model_student_4f_{student_name}.joblib')
with open(ARTIFACT_DIR / f'model_card_student_4f_{student_name}.json', 'w') as f:
    json.dump({
        'model': f'Student ({"XGBoost" if XGB_AVAILABLE else "LogisticRegression"})',
        'features_expected': {
            'sex': "['Male','Female']",
            'weight': 'float (kg)',
            'SMK_stat_type_cd': 'int [1=never,2=former,3=current]',
            'drinker_degree': 'int [0,1]'
        },
        'notes': ['Student distilled from teacher; inference needs only 4 inputs.']
    }, f, indent=2)
print('✅ Student saved to artifacts.')

# %% [markdown]
# ## 7. Inference Examples

# Choose whichever model you want to serve
TEACHER_PATH = ARTIFACT_DIR / f'risk_model_teacher_{teacher_name}.joblib'
STUDENT_PATH = ARTIFACT_DIR / f'risk_model_student_4f_{student_name}.joblib'

# --- Full model inference (all features) ---
def predict_risk_full(**kwargs):
    model = joblib.load(TEACHER_PATH)
    # Ensure all teacher features are present (fill missing with NaN -> imputed by pipeline)
    payload = {**{c: np.nan for c in (categorical_features + numeric_features)}, **kwargs}
    X_inf = pd.DataFrame([payload])
    pred = model.predict(X_inf)[0]
    proba = model.predict_proba(X_inf)[0]
    return pred, dict(zip(model.classes_, proba))

# --- Minimal 4‑feature inference ---
def predict_risk_minimal(sex: str, weight: float, smoker_degree: int, drinker_degree: int):
    model = joblib.load(STUDENT_PATH)
    X_inf = pd.DataFrame([{
        'sex': sex,
        'weight': weight,
        'SMK_stat_type_cd': smoker_degree,
        'drinker_degree': drinker_degree
    }])
    pred = model.predict(X_inf)[0]
    proba = model.predict_proba(X_inf)[0]
    return pred, dict(zip(model.classes_, proba))

# Examples
full_pred, full_proba = predict_risk_full(
    sex='Female', age=40, height=165, weight=60, waistline=75, SBP=120, DBP=80,
    tot_chole=180, HDL_chole=60, LDL_chole=110, triglyceride=100, gamma_GTP=25,
    hemoglobin=14, serum_creatinine=0.8, SGOT_AST=25, SGOT_ALT=20,
    SMK_stat_type_cd=1, drinker_degree=0
)
print('[Teacher] Pred:', full_pred); print('Proba:', full_proba)

stud_pred, stud_proba = predict_risk_minimal('Male', 82.0, 3, 1)
print('\n[Student 4f] Pred:', stud_pred); print('Proba:', stud_proba)

# %% [markdown]
# ## 8. Explainability (quick view)
# For tree models (RF/XGB), built-in feature importance (global). For the student LogReg, coefficients.

try:
    # Extract preprocessed feature names
    ohe = teacher_pipe.named_steps['preprocess'].named_transformers_['cat'].named_steps['ohe']
    cat_names = ohe.get_feature_names_out(categorical_features)
    num_names = numeric_features
    feat_names = np.concatenate([cat_names, num_names])

    clf = teacher_pipe.named_steps['clf']
    if XGB_AVAILABLE:
        importances = clf.feature_importances_
    else:
        importances = clf.feature_importances_

    imp_df = pd.DataFrame({'feature': feat_names, 'importance': importances}).sort_values('importance', ascending=False).head(25)
    plt.figure(figsize=(8,8))
    sns.barplot(data=imp_df, x='importance', y='feature')
    plt.title('Top Feature Importances (Teacher)')
    plt.tight_layout(); plt.show()
except Exception as e:
    print('Explainability plot skipped:', e)

# %% [markdown]
# ## 9. Conclusions & Next Steps
# - The **teacher** (XGBoost/RandomForest) leverages **all clinical features** and provides higher discriminative power.
# - The **student** is distilled to use **only 4 inputs** for deployment, achieving good agreement with the teacher.
# - Replace the proxy target with **real actuarial labels** for pricing/underwriting.
# - Add **calibration** (e.g., isotonic) and **fairness checks** by sex/age.
# - Monitor drift of input distributions and risk share over time.
