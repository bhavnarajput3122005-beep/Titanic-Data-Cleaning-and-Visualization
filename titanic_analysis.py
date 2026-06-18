import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("titanic.csv")

print("First 5 rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

# Clean data
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

if 'Cabin' in df.columns:
    df.drop('Cabin', axis=1, inplace=True)

df.drop_duplicates(inplace=True)

# Visualization
sns.countplot(x='Survived', data=df)
plt.title("Survival Count")
plt.show()


# Survival by Gender
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title("Survival by Gender")
plt.show()

# Age Distribution
sns.histplot(df['Age'], bins=20)
plt.title("Age Distribution")
plt.show()

# Passenger Class Distribution
sns.countplot(x='Pclass', data=df)
plt.title("Passenger Class Distribution")
plt.show()