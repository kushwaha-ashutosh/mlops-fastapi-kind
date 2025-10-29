# app/model_train.py

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import joblib

# ✅ Sample data: [study_hours, sleep_hours]
X = np.array([
    [1, 8], [2, 7], [3, 6], [4, 6], [5, 5],
    [6, 5], [7, 4], [8, 4], [9, 3], [10, 3]
])
# Target: 0 = fail, 1 = pass
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# ✅ Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc*100:.2f}%")

# ✅ Save model
joblib.dump(model, "app/model.pkl")
print("✅ Model saved to app/model.pkl")
