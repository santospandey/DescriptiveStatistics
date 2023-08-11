import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats


# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wpbc.data"
column_names = ["ID", "Outcome", "RadiusMean", "TextureMean", "PerimeterMean", "AreaMean"]
data = pd.read_csv(url, names=column_names)

# Numeric attributes for analysis
numeric_attributes = ["Outcome", "RadiusMean", "TextureMean", "PerimeterMean"]

data = pd.read_csv(url, names=column_names)
data.head(5)
data.describe()

# Histograms
data[numeric_attributes].hist(bins=20, figsize=(12, 10))
plt.tight_layout()
plt.show()

# Boxplot
for attribute in numeric_attributes:
    plt.boxplot(data[attribute])
    plt.title(attribute)
    plt.show()

# Steam plots
for attribute in numeric_attributes:
    plt.figure(figsize=(8, 6))
    plt.title(f"Stem Plot for {attribute}")
    plt.stem(data[attribute], use_line_collection=True)
    plt.xlabel("Index")
    plt.ylabel(attribute)
    plt.show()

# Q-Q plots
for attribute in numeric_attributes:
    sm.qqplot(data[attribute], line='s', fit=True, dist=stats.norm)
    plt.title(f"Q-Q Plot for {attribute}")
    plt.show()

## Probability plot
for attribute in numeric_attributes:    
    stats.probplot(data[attribute], dist="norm", fit=True, rvalue=True, plot=plt)
    plt.xlabel("", labelpad=15)
    plt.title(attribute, y=1.015)
    plt.show()
