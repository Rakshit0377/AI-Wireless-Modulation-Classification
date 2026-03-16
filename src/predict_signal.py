import joblib
import pandas as pd

model = joblib.load("models/signal_classifier.pkl")

# example signal
new_signal = pd.DataFrame([[0.4, 1.0, 0.6]], columns=["feature1","feature2","feature3"])

prediction = model.predict(new_signal)

print("Predicted Signal Type:", prediction[0])