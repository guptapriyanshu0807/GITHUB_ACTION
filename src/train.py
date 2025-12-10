import pandas as pd
import numpy as np
import os
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import json

#load the iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

#split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#save the model
joblib.dump(model, 'model.joblib')

#calculate and save matrics
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

metrics = {
    'train_accuracy': float(train_score),
    'test_accuracy': float(test_score)
}

with open('metrics.json', 'w') as f:
    json.dump(metrics, f)
    
print(f"Training accuracy : {train_score:.4f}")
print(f"Testing accuracy : {test_score:.4f}")