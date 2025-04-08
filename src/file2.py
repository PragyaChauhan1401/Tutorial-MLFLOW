import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import dagshub

dagshub.init(repo_owner='pragyac156', repo_name='Tutorial-MLFLOW', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/pragyac156/Tutorial-MLFLOW.mlflow")
# Load Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Define parameters
max_depth = 5
n_estimators = 10

# Set experiment

with mlflow.start_run():

    # Train model
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('n_estimators', n_estimators)

    # Confusion matrix (ensure it is np.array with int type)
    cm = np.array(confusion_matrix(y_test, y_pred), dtype=int)
    
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=wine.target_names.tolist(),
                yticklabels=wine.target_names.tolist())
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

    # Save and log confusion matrix image
    plot_path = "confusion_matrix.png"
    plt.savefig(plot_path)
    plt.close()

    mlflow.log_artifact(plot_path)

    # Safe logging of script
    mlflow.log_artifact(__file__)

    mlflow.set_tags({'Author': 'Pragya','Project':'Wine classification'})

    mlflow.sklearn.log_model(rf,'Random-forest-model')
    print(f"Accuracy: {accuracy:.4f}")
