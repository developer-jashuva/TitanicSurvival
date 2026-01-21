import pandas as pd
import matplotlib.pyplot as plt


#LOADING DATA
df =pd.read_csv('titanic.csv')

#HANDLING MISSING VALUES
df['Age']=df['Age'].fillna(df['Age'].median())
df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Cabin']=df['Cabin'].fillna('Unknown')
#df['Cabin'].isnull().sum()
#df['Age'].isnull().sum()
#df['Embarked'].isnull().sum()


#FILTERED THE RAW DATA

#df[['Age','Name']]
#df[df['Age']>18]
#df.loc[0:4, ["Age",'Survived','Sex']]
#df[(df['Sex']=="female") & (df['Survived']==1)]

#FINDING INSIGHTS
df["Survived"].value_counts()
df.groupby("Sex")["Survived"].mean()
df.groupby("Pclass")["Survived"].mean()
df.groupby(["Sex", "Pclass"])["Survived"].mean()

df["AgeGroup"] = pd.cut(
    df["Age"],
    bins=[0, 12, 18, 40, 60, 100],
    labels=["Child", "Teen", "Adult", "MiddleAge", "Senior"]
)
df.groupby("AgeGroup")["Survived"].mean()

df[["Age", "AgeGroup"]]


#VISUAL REPRESENTATION

#Overall Survival Rate
df["Survived"].value_counts().plot(kind="bar")
plt.title("Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")
plt.show()

#Survival Rate By Gender
df.groupby("Sex")["Survived"].mean().plot(kind="bar")
plt.title("Survival Rate by Gender")
plt.ylabel("Survival Probability")
plt.show()

#Survival Rate By Passanger Class
df.groupby("Pclass")["Survived"].mean().plot(kind="bar")
plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Probability")
plt.show()

#Survival Rate By Age Group
df.groupby("AgeGroup")["Survived"].mean().plot(kind="bar")
plt.title("Survival Rate by Age Group")
plt.ylabel("Survival Probability")
plt.show()
