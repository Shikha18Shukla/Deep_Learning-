#load tha california housing dataset and create a DataFrame with features and target

import pandas as pd

df = pd.read_csv("housingdatset.csv")
selected = df[[
    "total_rooms", 
    "total_bedrooms", 
    "population",
    "median_house_value", 
    "ocean_proximity"
]]

print(df.head())
print(df.info())
#print(selected.head())
#print(selected.info())


'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("housingdatset.csv")

selected = df[[
    "total_rooms", 
    "total_bedrooms", 
    "population", 
    "households", 
    "median_income", 
    "median_house_value", 
    "ocean_proximity"
]]


selected.hist(bins=50, figsize=(12, 8))
plt.suptitle("Histograms of Numeric Features")
plt.show()


plt.figure(figsize=(12, 6))
sns.boxplot(data=selected.drop("ocean_proximity", axis=1))
plt.title("Boxplot of Numeric Features")
plt.show()


plt.figure(figsize=(10, 6))
sns.heatmap(selected.drop("ocean_proximity", axis=1).corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


plt.figure(figsize=(8, 5))
sns.countplot(x="ocean_proximity", data=selected)
plt.title("Count of Ocean Proximity Categories")
plt.show()


plt.figure(figsize=(8, 6))
sns.scatterplot(x="median_income", y="median_house_value", data=selected, hue="ocean_proximity")
plt.title("Income vs House Value")
plt.show()
'''