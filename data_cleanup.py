# data analysis and wrangling
import pandas as pd
import numpy as np

def get_data():

    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')

    # Data Clean Up
    # --------------------------------------------------------------------------------------------------

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
    train['Fare'].fillna(train['Fare'].dropna().mean(), inplace=True)
    test['Fare'].fillna(test['Fare'].dropna().mean(), inplace=True)

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
    train['Embarked'] = train['Embarked'].fillna("S")
    train['Embarked'] = train['Embarked'].map( {'C': 0, 'Q': 1, 'S': 2} ).astype(int)
    test['Embarked'] = test['Embarked'].fillna("S")
    test['Embarked'] = test['Embarked'].map( {'C': 0, 'Q': 1, 'S': 2} ).astype(int)

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

    # Create alone varibles
    train['IsAlone'] = 0
    train.loc[train['FamilyMems'] == 1, 'IsAlone'] = 1
    test['IsAlone'] = 0
    test.loc[test['FamilyMems'] == 1, 'IsAlone'] = 1

    ids = test['PassengerId']
    train = train[["Pclass", "Sex", "AgeRange", "Title", 'CabinLetter', 'Embarked', 'FamilyMems', 'Fare', 'IsAlone', "Survived"]]
    test = test[["Pclass", "Sex", "AgeRange", "Title", 'CabinLetter', 'Embarked', 'FamilyMems', 'Fare', 'IsAlone']]

    return (train, test, ids)


def get_dummy_data():
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')

    # Data Clean Up
    # --------------------------------------------------------------------------------------------------

    # Columns that do not seem important: PassengerID, Name, Ticket

    # Convert sex to 0/1
    train['Sex'] = train['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
    test['Sex'] = test['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

    # Get Cabin letter from Cabin and convert to number
    train['Cabin'] = train.Cabin.str.extract('([A-Za-z])', expand=False)
    train['Cabin'] = train['Cabin'].fillna("NA")
    test['Cabin'] = test.Cabin.str.extract('([A-Za-z])', expand=False)
    test['Cabin'] = test['Cabin'].fillna("NA")
    train = pd.get_dummies(train, columns = ['Cabin'])
    test = pd.get_dummies(test, columns = ['Cabin'])
    test["Cabin_T"] = 0

    # Clean up NAs from fare
    train['Fare'].fillna(train['Fare'].dropna().mean(), inplace=True)
    test['Fare'].fillna(test['Fare'].dropna().mean(), inplace=True)

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
    train.loc[ train['Age'] <= 6, 'Age'] = 0
    train.loc[(train['Age'] > 6) & (train['Age'] <= 21), 'Age'] = 1
    train.loc[(train['Age'] > 21) & (train['Age'] <= 26), 'Age'] = 2
    train.loc[(train['Age'] > 26) & (train['Age'] <= 36), 'Age'] = 3
    train.loc[(train['Age'] > 36), 'Age'] = 4

    test.loc[ test['Age'] <= 6, 'Age'] = 0
    test.loc[(test['Age'] > 6) & (test['Age'] <= 21), 'Age'] = 1
    test.loc[(test['Age'] > 21) & (test['Age'] <= 26), 'Age'] = 2
    test.loc[(test['Age'] > 26) & (test['Age'] <= 36), 'Age'] = 3
    test.loc[(test['Age'] > 36), 'Age'] = 4

    train = pd.get_dummies(train, columns = ['Age'])
    test = pd.get_dummies(test, columns = ['Age'])

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
    test['Title'] = test['Title'].fillna("Unk")

    train = pd.get_dummies(train, columns = ['Title'])
    test = pd.get_dummies(test, columns = ['Title'])

    # Convert Embarked to numbers
    train['Embarked'] = train['Embarked'].fillna("S")
    test['Embarked'] = test['Embarked'].fillna("S")
    train = pd.get_dummies(train, columns = ['Embarked'])
    test = pd.get_dummies(test, columns = ['Embarked'])
    test["Embarked_NA"] = 0

    # Find number of family members on board
    train['FamilyMems'] = train['Parch'] + train['SibSp']
    test['FamilyMems'] = test['Parch'] + test['SibSp']
    train = pd.get_dummies(train, columns = ['FamilyMems'])
    test = pd.get_dummies(test, columns = ['FamilyMems'])

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

    train = pd.get_dummies(train, columns = ['Fare'])
    test = pd.get_dummies(test, columns = ['Fare'])
    train = pd.get_dummies(train, columns = ['Pclass'])
    test = pd.get_dummies(test, columns = ['Pclass'])
    ids = test['PassengerId']

    train = train.drop(["PassengerId", "Name", "SibSp", "Parch", "Ticket"], axis=1)
    test = test.drop(["PassengerId", "Name", "SibSp", "Parch", "Ticket"], axis=1)

    return (train, test, ids)
