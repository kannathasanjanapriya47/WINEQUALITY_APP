import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv("data/winequality-red.csv")

# Convert quality score Good (1) or Bad (0
df['quality_label'] = df['quality'].apply(lambda x: 1 if x >= 7 else 0)

# Features (X) and target (y)
X = df.drop(['quality', 'quality_label'], axis=1)
y = df['quality_label']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models  that we can compare
models = {
    'logreg': LogisticRegression(max_iter=1000),
    'rf': RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}
for name, m in models.items():
    # Cross-validation score
    scores = cross_val_score(m, X_train, y_train, cv=5, scoring='accuracy')
    m.fit(X_train, y_train)
    y_pred = m.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = {'cv_mean': scores.mean(), 'test_acc': acc}
    print(f"{name} → CV Mean: {scores.mean():.4f}, Test Accuracy: {acc:.4f}")

# Choose the best model
best_name = max(results, key=lambda k: results[k]['test_acc'])
best_model = models[best_name]
print(f" Best model: {best_name} with accuracy {results[best_name]['test_acc']:.4f}")

# Save model, feature names, and target names
joblib.dump({
    'model': best_model,
    'feature_names': X.columns.tolist(),
    'target_names': ['Bad', 'Good']
}, 'model.pkl')

print("Model saved as model.pkl")

