# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# Columns that do not seem important: PassengerID, Name, Ticket

# Convert sex to 0/1
train['Sex'] = train['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
test['Sex'] = test['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

# Get Cabin letter from Cabin and convert to number
train['CabinLetter'] = train.Cabin.str.extract('([A-Za-z])', expand=False)
train['CabinLetter'] = train['CabinLetter'].fillna("NA")
train['CabinLetter'] = train['CabinLetter'].map( {'D': 1, 'E': 2, 'B': 3, 'F': 4, 'C': 5, 'G': 6, 'A': 7, 'NA': 8, 'T': 9} ).astype(int)
test['CabinLetter'] = test.Cabin.str.extract('([A-Za-z])', expand=False)
test['CabinLetter'] = test['CabinLetter'].fillna("NA")
test['CabinLetter'] = test['CabinLetter'].map( {'D': 1, 'E': 2, 'B': 3, 'F': 4, 'C': 5, 'G': 6, 'A': 7, 'NA': 8, 'T': 9} ).astype(int)

# Clean up NAs from fare
train['Fare'].fillna(train['Fare'].dropna().median(), inplace=True)
test['Fare'].fillna(test['Fare'].dropna().median(), inplace=True)

# Clean up NAs from age
guess_ages = np.zeros((2,3))
guess_ages_test = np.zeros((2,3))
for i in range(0, 2):
    for j in range(0, 3):
        guess_df = train[(train['Sex'] == i) & (train['Pclass'] == j+1)]['Age'].dropna()
        guess_df2 = test[(test['Sex'] == i) & (test['Pclass'] == j+1)]['Age'].dropna()

        age_guess = guess_df.median()
        age_guess_test = guess_df2.median()

        # Convert random age float to nearest .5 age
        guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
        guess_ages_test[i,j] = int( age_guess_test/0.5 + 0.5 ) * 0.5

for i in range(0, 2):
    for j in range(0, 3):
        train.loc[(train.Age.isnull()) & (train.Sex == i) & (train.Pclass == j+1), 'Age'] = guess_ages[i,j]
        test.loc[(test.Age.isnull()) & (test.Sex == i) & (test.Pclass == j+1), 'Age'] = guess_ages_test[i,j]

train['Age'] = train['Age'].astype(int)
test['Age'] = test['Age'].astype(int)

# Convert Age into ranges
train['AgeRange'] = train['Age']
train.loc[ train['AgeRange'] <= 6, 'AgeRange'] = 0
train.loc[(train['AgeRange'] > 6) & (train['AgeRange'] <= 21), 'AgeRange'] = 1
train.loc[(train['AgeRange'] > 21) & (train['AgeRange'] <= 26), 'AgeRange'] = 2
train.loc[(train['AgeRange'] > 26) & (train['AgeRange'] <= 36), 'AgeRange'] = 3
train.loc[(train['AgeRange'] > 36), 'AgeRange'] = 4

test['AgeRange'] = test['Age']
test.loc[ test['AgeRange'] <= 6, 'AgeRange'] = 0
test.loc[(test['AgeRange'] > 6) & (test['AgeRange'] <= 21), 'AgeRange'] = 1
test.loc[(test['AgeRange'] > 21) & (test['AgeRange'] <= 26), 'AgeRange'] = 2
test.loc[(test['AgeRange'] > 26) & (test['AgeRange'] <= 36), 'AgeRange'] = 3
test.loc[(test['AgeRange'] > 36), 'AgeRange'] = 4

# Get titles from name
train['Title'] = train.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
train['Title'] = train['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
train['Title'] = train['Title'].replace('Mlle', 'Miss')
train['Title'] = train['Title'].replace('Ms', 'Miss')
train['Title'] = train['Title'].replace('Mme', 'Mrs')
test['Title'] = test.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
test['Title'] = test['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
test['Title'] = test['Title'].replace('Mlle', 'Miss')
test['Title'] = test['Title'].replace('Ms', 'Miss')
test['Title'] = test['Title'].replace('Mme', 'Miss')

# Convert titles to numbers
train['Title'] = train['Title'].fillna("Unk")
train['Title'] = train['Title'].map( {'Unk': -1, 'Rare': 0, 'Miss': 1, 'Master': 2, 'Mr': 3, 'Mrs': 4} ).astype(int)
test['Title'] = test['Title'].fillna("Unk")
test['Title'] = test['Title'].map( {'Unk': -1, 'Rare': 0, 'Miss': 1, 'Master': 2, 'Mr': 3, 'Mrs': 4} ).astype(int)

# Convert Embarked to numbers
train['Embarked'] = train['Embarked'].fillna("NA")
train['Embarked'] = train['Embarked'].map( {'NA': -1, 'C': 0, 'Q': 1, 'S': 2} ).astype(int)
test['Embarked'] = test['Embarked'].fillna("NA")
test['Embarked'] = test['Embarked'].map( {'NA': -1, 'C': 0, 'Q': 1, 'S': 2} ).astype(int)

# Find number of family members on board
train['FamilyMems'] = train['Parch'] + train['SibSp']
test['FamilyMems'] = test['Parch'] + test['SibSp']

# Get fare ranges
train.loc[ train['Fare'] <= 7.91, 'Fare'] = 0
train.loc[(train['Fare'] > 7.91) & (train['Fare'] <= 14.454), 'Fare'] = 1
train.loc[(train['Fare'] > 14.454) & (train['Fare'] <= 31), 'Fare']   = 2
train.loc[ train['Fare'] > 31, 'Fare'] = 3
train['Fare'] = train['Fare'].astype(int)
test.loc[ test['Fare'] <= 7.91, 'Fare'] = 0
test.loc[(test['Fare'] > 7.91) & (test['Fare'] <= 14.454), 'Fare'] = 1
test.loc[(test['Fare'] > 14.454) & (test['Fare'] <= 31), 'Fare']   = 2
test.loc[ test['Fare'] > 31, 'Fare'] = 3
test['Fare'] = test['Fare'].astype(int)


# Embarked could be useful
train[['FamilyMems', 'Survived']].groupby(['FamilyMems'], as_index=False).mean().sort_values(by='Survived', ascending=False)


grid = sns.FacetGrid(train, col='Survived', row='CabinLetter', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Fare', alpha=.5, bins=20)
grid.add_legend();


train.head()

X_train = train.drop("Survived", axis=1)
Y_train = train["Survived"]
X_test  = test.drop("PassengerId", axis=1).copy()
ids = test['PassengerId']

# Starting with a simple log regresssion
logreg = LogisticRegression()
logreg.fit(X_train[["Pclass", "Sex", "AgeRange", "Title", 'CabinLetter', 'Embarked', 'FamilyMems', 'Fare']], Y_train)
Y_pred = logreg.predict(X_test[["Pclass", "Sex", "AgeRange", "Title", 'CabinLetter', 'Embarked', 'FamilyMems', 'Fare']])
acc_log = round(logreg.score(X_train[["Pclass", "Sex", "AgeRange", "Title", 'CabinLetter', 'Embarked', 'FamilyMems', 'Fare']], Y_train) * 100, 2)
acc_log

coeff_df = pd.DataFrame(X_train[["Pclass", "Sex", "AgeRange", "Title", 'CabinLetter', 'Embarked', 'FamilyMems', 'Fare']].columns)
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)

# Using Random Forest



output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': Y_pred })
output.to_csv('titanic-predictions.csv', index = False)
