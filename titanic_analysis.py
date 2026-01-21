import pandas as pd
df =pd.read_csv('titanic.csv')

df['Age']=df['Age'].fillna(df['Age'].median())
#df['Age'].isnull().sum()

df['Cabin']=df['Cabin'].fillna('Unknown')
#df['Cabin'].isnull().sum()


df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])
#df['Embarked'].isnull().sum()

# df.isnull().sum()
# df.info()
#df[['Age','Name']]
#df[df['Age']>18]
#df.loc[0:4, ["Age",'Survived','Sex']]
#df[(df['Sex']=="female") & (df['Survived']==1)]

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

df[["Age", "AgeGroup"]].head(10)
