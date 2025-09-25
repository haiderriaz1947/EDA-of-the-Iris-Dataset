# loading library 
import pandas as pd
import numpy as py
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.express as px
import plotly.graph_objects as go

# Load the Iris dataset using scikit-learn or seaborn
from sklearn.datasets import load_iris
import pandas as pd
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target

df = sns.load_dataset('iris')

import pandas as pd 
df = pd.read_csv("Iris.csv")

df.head()

print(df.shape)

print(df.info())

print(df.isnull().sum())

print(df.describe())

print(df.groupby('Species_name').mean())

print(df.groupby('Species_name').std())

import matplotlib.pyplot as plt
columns = df.select_dtypes(include='number').columns
axes = df.hist(figsize=(10, 8), grid=False)
for ax, col in zip(axes.flatten(), columns):
    ax.set_title(f'Distribution of {col}', fontsize=12)
    ax.set_xlabel(f'{col} (cm)')
    ax.set_ylabel('Frequency')
plt.tight_layout()
plt.show()

import plotly.express as px
fig = px.scatter(df, x='Sepal_length', y='Species_name', title='Sepal Length vs Target Classes in Iris Dataset')
fig.show()

import plotly.express as px
fig = px.scatter(df, x='Petal_length', y='Petal_width', color='Species_name', title='Petal Length vs Petal Width')
fig.show()

import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
sns.boxplot(x='Species_name', y='Petal_length', data=df)
plt.title('Boxplot of Petal Length by Species')
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
sns.kdeplot(data=df, x='Petal_length', hue='Species_name', fill=True)
plt.title('KDE Plot of Petal Length by Species')
plt.show()

import seaborn as sns
sns.pairplot(df, hue='Species_name')
plt.show()

import plotly.express as px
fig = px.scatter(df, x='Petal_length', y='Petal_width', trendline='ols', title='Petal Length vs Petal Width with Regression Line')
fig.show()

import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
correlation_matrix = df.select_dtypes(include='number').corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix Heatmap')
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
sns.violinplot(x='Species_name', y='Petal_length', data=df)
plt.title('Violin Plot of Petal Length by Species')
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
sns.stripplot(x='Species_name', y='Petal_length', data=df, jitter=True)
plt.title('Strip Plot of Petal Length by Species')
plt.show()

import seaborn as sns
g = sns.FacetGrid(df, col='Species_name')
g.map(plt.scatter, 'Petal_length', 'Petal_width')
plt.show()

import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
df.hist(ax=axes)
plt.show()

pivot_table = df.pivot_table(index='Species_name', values=['Petal_length', 'Petal_width'], aggfunc='mean')
print(pivot_table)

pivot_table_multi = df.pivot_table(index='Species_name', values=['Petal_length', 'Petal_width'], aggfunc=['mean', 'std'])
print(pivot_table_multi)

import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
sns.heatmap(pivot_table, annot=True, cmap='YlGnBu')
plt.title('Heatmap of Mean Petal Length and Width by Species')
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.show()
