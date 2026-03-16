import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data/processed/dataset.csv")

plt.scatter(data["feature1"], data["feature2"], c=data["label"].astype('category').cat.codes)

plt.title("Signal Feature Distribution")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.show()