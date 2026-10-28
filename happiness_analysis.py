import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("happiness-report-2024.csv")

print("\nGLOBAL HAPPINESS REPORT ANALYSIS")
print("----------------------------------")

print("\nFirst 5 rows of the dataset:")
print(data.head())

print("\nDataset Information:")
print(data.info())

print("\nNumber of rows and columns:")
print(data.shape)

print("\nMissing Values:")
print(data.isnull().sum())

data = data.dropna(
    subset=[
        "Country name",
        "Ladder score",
        "Log GDP per capita",
        "Social support",
        "Healthy life expectancy"
    ]
)

print("\nStatistical Summary:")
print(data.describe())

top_10 = data.sort_values(
    by="Ladder score",
    ascending=False
).head(10)

print("\nTop 10 Happiest Countries:")
print(top_10[["Country name", "Ladder score"]])

bottom_10 = data.sort_values(
    by="Ladder score",
    ascending=True
).head(10)

print("\n10 Least Happy Countries:")
print(bottom_10[["Country name", "Ladder score"]])

average_happiness = data["Ladder score"].mean()

print("\nAverage Happiness Score:")
print(round(average_happiness, 2))

plt.figure(figsize=(10, 6))

sns.barplot(
    data=top_10,
    x="Ladder score",
    y="Country name",
    hue="Country name",
    legend=False,
    palette="viridis"
)

plt.title("Top 10 Happiest Countries")
plt.xlabel("Happiness Score")
plt.ylabel("Country")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))

sns.regplot(
    data=data,
    x="Social support",
    y="Ladder score",
    scatter_kws={"color": "cyan", "s": 60, "alpha": 0.7},
    line_kws={"color": "purple", "linewidth": 2}
)

plt.title("GDP Per Capita vs Happiness")
plt.xlabel("Log GDP Per Capita")
plt.ylabel("Happiness Score")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))

sns.kdeplot(
    data=data,
    x="Social support",
    y="Ladder score",
    fill=True,
    cmap="rocket",
    levels=10,
    thresh=0.05
)

plt.title("Social Support vs Happiness Density", fontsize=16)
plt.xlabel("Social Support")
plt.ylabel("Happiness Score")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))

plt.hexbin(
    data=data,
    x="Healthy life expectancy",
    y="Ladder score",
    gridsize=20,
    cmap="crest",
    mincnt=1
)

plt.colorbar(label="Number of countries")

plt.title("Life Expectancy vs Happiness")
plt.xlabel("Healthy Life Expectancy")
plt.ylabel("Happiness Score")
plt.tight_layout()
plt.show()

factors = [
    "Ladder score",
    "Log GDP per capita",
    "Social support",
    "Healthy life expectancy",
    "Freedom to make life choices",
    "Generosity",
    "Perceptions of corruption"
]

correlation = data[factors].corr()

plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation,
    annot=True,
    cmap="flare",
    fmt=".2f"
)

plt.title("Correlation Between Happiness Factors")
plt.tight_layout()
plt.show()

print("\nFINAL RESULTS")
print("----------------------------------")

print("Total Countries Analyzed:", len(data))

print("Average Happiness Score:", round(average_happiness, 2))

print(
    "Happiest Country:",
    top_10.iloc[0]["Country name"]
)

print(
    "Highest Happiness Score:",
    round(top_10.iloc[0]["Ladder score"], 2)
)

print("\nAnalysis Completed Successfully!")