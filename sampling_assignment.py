import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import numpy as np

data = pd.read_csv("Creditcard_data.csv")
print("Dataset successfully loaded.")

print(data.info())

X = data.drop(columns=["Class"])
y = data["Class"]

print("Balancing dataset...")
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

print("Original class distribution:")
print(y.value_counts())
print("Balanced class distribution:")
print(y_balanced.value_counts())

sample_size = min(len(X_balanced), 1000)
print(f"Generating samples with size: {sample_size}")
samples = {
    f"Sample_{i+1}": X_balanced.sample(n=sample_size, random_state=seed)
    for i, seed in enumerate([42, 21, 56, 99, 77])
}

sample_data = {
    name: (sample, y_balanced.loc[sample.index])
    for name, sample in samples.items()
}

print("Initializing classifiers...")
models = {
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
    "ExtraTrees": ExtraTreesClassifier(random_state=42),
    "SVC": SVC(random_state=42),
    "RandomForest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42, verbosity=0),
}

print("Training and evaluating models...")
results = {}
for sample_name, (X_sample, y_sample) in sample_data.items():
    print(f"Processing {sample_name}...")
    X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.3, random_state=42)

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        if sample_name not in results:
            results[sample_name] = {}
        results[sample_name][model_name] = accuracy

print("Summarizing results...")
results_matrix = pd.DataFrame(results).T

results_matrix.to_csv("model_results.csv")
print("Model results saved to 'model_results.csv'.")

best_methods = results_matrix.idxmax()
best_methods.to_csv("best_methods.csv")

print("Best sampling techniques for each model:")
print(best_methods)
print("Summary saved to 'best_methods.csv'.")
