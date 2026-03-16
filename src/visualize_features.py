import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("data/processed/dataset.csv")

sns.pairplot(data, hue="label")

plt.show()