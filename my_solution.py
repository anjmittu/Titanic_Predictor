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
train['CabinLetter'] = train['CabinLetter'].map( {'NA': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 8} ).astype(int)
test['CabinLetter'] = test.Cabin.str.extract('([A-Za-z])', expand=False)
test['CabinLetter'] = test['CabinLetter'].fillna("NA")
test['CabinLetter'] = test['CabinLetter'].map( {'NA': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 8} ).astype(int)

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

        # age_mean = guess_df.mean()
        # age_std = guess_df.std()
        # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

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
train.loc[ train['AgeRange'] <= 21, 'AgeRange'] = 0
train.loc[(train['AgeRange'] > 21) & (train['AgeRange'] <= 26), 'AgeRange'] = 1
train.loc[(train['AgeRange'] > 26) & (train['AgeRange'] <= 36), 'AgeRange'] = 2
train.loc[(train['AgeRange'] > 36), 'AgeRange'] = 3

test['AgeRange'] = test['Age']
test.loc[ train['AgeRange'] <= 21, 'AgeRange'] = 0
test.loc[(train['AgeRange'] > 21) & (test['AgeRange'] <= 26), 'AgeRange'] = 1
test.loc[(train['AgeRange'] > 26) & (test['AgeRange'] <= 36), 'AgeRange'] = 2
test.loc[(train['AgeRange'] > 36), 'AgeRange'] = 3



# Embarked could be useful
train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# Cabin letter could be useful
train[['CabinLetter', 'Survived']].groupby(['CabinLetter'], as_index=False).mean().sort_values(by='Survived', ascending=False)



grid = sns.FacetGrid(train, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Fare', alpha=.5, bins=20)
grid.add_legend();


train.head()

X_train = train.drop("Survived", axis=1)
Y_train = train["Survived"]
X_test  = test.drop("PassengerId", axis=1).copy()
ids = test['PassengerId']
combine = [train, test]

# Starting with a simple log regresssion
logreg = LogisticRegression()
logreg.fit(X_train[["Pclass", "Sex", "AgeRange"]], Y_train)
Y_pred = logreg.predict(X_test[["Pclass", "Sex", "AgeRange"]])
acc_log = round(logreg.score(X_train[["Pclass", "Sex", "AgeRange"]], Y_train) * 100, 2)
acc_log


output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': Y_pred })
output.to_csv('titanic-predictions.csv', index = False)
output.head()
