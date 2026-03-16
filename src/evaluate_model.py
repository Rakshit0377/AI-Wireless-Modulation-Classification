import joblib
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# load dataset
data = pd.read_csv("data/processed/dataset.csv")

X = data.drop("label", axis=1)
y = data["label"]

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# load trained model
model = joblib.load("models/signal_classifier.pkl")

# predictions
pred = model.predict(X_test)

print("Classes:", model.classes_)
print()
print(classification_report(y_test, pred, zero_division=0))
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

ConfusionMatrixDisplay.from_predictions(y_test, pred)
plt.show()