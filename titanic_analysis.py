import pandas as pd
import matplotlib.pyplot as plt


# 1. Load Dataset
df =pd.read_csv('titanic.csv')

# 2. Handle Missing Values
df['Age']=df['Age'].fillna(df['Age'].median())
df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Cabin']=df['Cabin'].fillna('Unknown')

# 3. Feature Engineering
df["AgeGroup"] = pd.cut(
    df["Age"],
    bins=[0, 12, 18, 40, 60, 100],
    labels=["Child", "Teen", "Adult", "MiddleAge", "Senior"]
)

# 4. Exploratory Data Analysis

#Overall Survival Rate

survival_count = df["Survived"].value_counts()

# Survival rate by gender
survival_by_gender = df.groupby("Sex")["Survived"].mean()

# Survival rate by passenger class
survival_by_class = df.groupby("Pclass")["Survived"].mean()

# Survival rate by gender and class
survival_gender_class = df.groupby(["Sex", "Pclass"])["Survived"].mean()

# Survival rate by age group
survival_by_agegroup = df.groupby("AgeGroup")["Survived"].mean()

# 5. Visualizations


# Overall Survival
survival_count.plot(kind="bar")
plt.title("Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")
plt.show()

# Survival by Gender
survival_by_gender.plot(kind="bar")
plt.title("Survival Rate by Gender")
plt.ylabel("Survival Probability")
plt.show()

# Survival by Passenger Class
survival_by_class.plot(kind="bar")
plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Probability")
plt.show()

# Survival by Age Group
survival_by_agegroup.plot(kind="bar")
plt.title("Survival Rate by Age Group")
plt.ylabel("Survival Probability")
plt.show()
