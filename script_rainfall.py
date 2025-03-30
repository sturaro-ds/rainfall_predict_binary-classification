import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve, auc, classification_report, accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from ydata_profiling import ProfileReport
from scipy.stats import zscore
import warnings

# Ignore warnings
warnings.filterwarnings('ignore')

# Load datasets
train = pd.read_csv('/Users/claudiosturaro/Sturaro/5_KAGGLE/015_Binary_Prediction_Rainfall_Dataset/train.csv')
test = pd.read_csv('/Users/claudiosturaro/Sturaro/5_KAGGLE/015_Binary_Prediction_Rainfall_Dataset/test.csv')
sub_id = test.copy()  # Copy test dataset for submission

# Remove 'id' column
train.drop(columns=['id'], inplace=True)
test.drop(columns=['id'], inplace=True)

# Convert target variable to categorical
train['rainfall'] = train['rainfall'].astype('category')

# Handle missing values
if test['winddirection'].isnull().sum() > 0:
    test['winddirection'].fillna(test['winddirection'].mean(), inplace=True)

# Selected features for visualization
cols = ['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint',
        'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']

# KDE plots
fig, axes = plt.subplots(2, 5, figsize=(18, 9))
axes = axes.flatten()
for i, col in enumerate(cols):
    sns.kdeplot(data=train, x=col, hue='rainfall', ax=axes[i])
    axes[i].set_title(col)
plt.tight_layout()
plt.show()

# Histogram plots
fig, axes = plt.subplots(2, 5, figsize=(18, 9))
axes = axes.flatten()
for i, col in enumerate(cols):
    sns.histplot(data=train, x=col, hue='rainfall', ax=axes[i])
    axes[i].set_title(col)
plt.tight_layout()
plt.show()

# Boxplots with Z-score normalization
z_train = pd.DataFrame(zscore(train[cols]), columns=cols)
z_train['rainfall'] = train['rainfall']
plt.figure(figsize=(10, 5))
sns.boxplot(data=z_train)
plt.xticks(rotation=85)
plt.tight_layout()
plt.show()

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(z_train[cols], z_train['rainfall'], test_size=0.2, random_state=42)

# Train baseline model (Logistic Regression)
base_model = LogisticRegressionCV()
base_model.fit(X_train, y_train)

# Feature importance (only for linear models)
if hasattr(base_model, 'coef_'):
    pd.DataFrame(base_model.coef_, columns=base_model.feature_names_in_).T.plot(kind='bar')
    plt.show()

# Evaluate baseline model
y_pred = base_model.predict(X_test)
y_probs = base_model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC AUC: {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title("Baseline Model ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

# Function to train multiple models and compare performance
def train_best_classifier(X, y, test_size=0.2, random_state=42, n_iter=30, cv=5):
    """
    Train multiple classifiers, optimize hyperparameters, evaluate performance, and plot ROC curves.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    models = {
        "RandomForest": RandomForestClassifier(random_state=random_state),
        "GradientBoosting": GradientBoostingClassifier(random_state=random_state),
        "LogisticRegression": LogisticRegression(solver='liblinear'),
        "SVC": SVC(probability=True, random_state=random_state),
        "XGBoost": XGBClassifier(random_state=random_state, use_label_encoder=False, eval_metric='logloss')
    }
    
    param_grids = {
        "RandomForest": {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20, 30]},
        "GradientBoosting": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2]},
        "LogisticRegression": {"C": [0.01, 0.1, 1, 10], "penalty": ["l1", "l2"]},
        "SVC": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
        "XGBoost": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2]}
    }
    
    results = []
    plt.figure(figsize=(10, 8))
    
    for name, model in models.items():
        search = RandomizedSearchCV(model, param_grids[name], n_iter=n_iter, cv=StratifiedKFold(), scoring='roc_auc', n_jobs=-1, random_state=random_state)
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        y_prob = best_model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_prob)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.2f})')
        results.append((name, auc_score, search.best_params_))
    
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for All Models')
    plt.legend()
    plt.show()
    
    return pd.DataFrame(results, columns=["Model", "AUC", "Best Params"]).sort_values(by="AUC", ascending=False)

# Train and evaluate models
results_df = train_best_classifier(train[cols], train["rainfall"])
print(results_df)

# Make predictions with best model
best_model = XGBClassifier().fit(X_train, y_train)
predictions = best_model.predict_proba(test)[:, 1]
pd.DataFrame({'id': sub_id['id'], 'rainfall': predictions}).to_csv('submission.csv', index=False)
