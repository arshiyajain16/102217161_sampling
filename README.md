# Credit Card Fraud Detection

This project focuses on detecting credit card fraud by utilizing machine learning classifiers and addressing class imbalance through advanced techniques like SMOTE (Synthetic Minority Oversampling Technique).

---

## Steps in the Code

### Step 1: Import Libraries
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import numpy as np
```

### Step 2: Load the Dataset
The dataset is loaded from `Creditcard_data.csv`.
```python
data = pd.read_csv("Creditcard_data.csv")
print("Dataset successfully loaded.")
```

### Step 3: Handle Class Imbalance

The dataset is often imbalanced, which can affect the performance of machine learning models. To address this, SMOTE (Synthetic Minority Oversampling Technique) is applied to balance the dataset by generating synthetic samples for the minority class.

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)
```

### Step 4: Create Data Samples

To evaluate the models effectively, five balanced samples are generated from the dataset using different random seeds.

```python
sample_size = min(len(X_balanced), 1000)
samples = {
    f"Sample_{i+1}": X_balanced.sample(n=sample_size, random_state=seed)
    for i, seed in enumerate([42, 21, 56, 99, 77])
}
```

### Step 5: Initialize Classifiers

Five machine learning models are initialized to evaluate the performance on the balanced samples:

1. **Gradient Boosting Classifier**
2. **Extra Trees Classifier**
3. **Support Vector Classifier (SVC)**
4. **Random Forest Classifier**
5. **XGBoost Classifier**

```python
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

models = {
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
    "ExtraTrees": ExtraTreesClassifier(random_state=42),
    "SVC": SVC(random_state=42),
    "RandomForest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42, verbosity=0),
}
```

### Step 6: Train and Evaluate Models

The balanced samples are used to train and evaluate the classifiers. Each sample is split into training and testing sets, and the accuracy of each classifier is recorded.

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

results = {}

for sample_name, (X_sample, y_sample) in sample_data.items():
    print(f"Processing {sample_name}...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_sample, y_sample, test_size=0.3, random_state=42
    )
    
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        if sample_name not in results:
            results[sample_name] = {}
        results[sample_name][model_name] = accuracy
```

### Step 7: Storing the Data in `.csv` File and Finding the Best Combinations

The results of the models' performance are saved in a `.csv` file for further analysis. Additionally, the best sampling technique for each model is identified.

```python
results_matrix = pd.DataFrame(results).T

results_matrix.to_csv("model_results.csv")
print("Model results saved to 'model_results.csv'.")

best_methods = results_matrix.idxmax()
print("Best sampling techniques for each model:")
print(best_methods)

best_methods.to_csv("best_methods.csv")
print("Summary saved to 'best_methods.csv'.")
```