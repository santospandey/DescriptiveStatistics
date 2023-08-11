import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
column_names = ["symboling", "normalized-losses", "make", "fuel-type", "aspiration", "num-of-doors",
                "body-style", "drive-wheels", "engine-location", "wheel-base", "length", "width",
                "height", "curb-weight", "engine-type", "num-of-cylinders", "engine-size", "fuel-system",
                "bore", "stroke", "compression-ratio", "horsepower", "peak-rpm", "city-mpg", "highway-mpg", "price"]

# Numeric attributes for analysis
numeric_attributes = ["wheel-base", "curb-weight", "engine-size", "city-mpg"]

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