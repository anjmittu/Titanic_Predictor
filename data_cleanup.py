# data analysis and wrangling
import pandas as pd
import numpy as np

def cleanTicket(ticket):
    ticket = ticket.replace('.','')
    ticket = ticket.replace('/','')
    ticket = ticket.split()
    ticket = list(map(lambda t : t.strip() , ticket))
    ticket = list(filter(lambda t : not t.isdigit(), ticket))
    if len(ticket) > 0:
        return ticket[0]
    else:
        return 'XXX'

def get_data():

    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    ids = test['PassengerId']

    # Combine the two data sets just to make sure each end with same set of features
    combined = train.append(test)
    combined.reset_index(inplace=True)
    combined.drop('index',inplace=True,axis=1)

    # Data Clean Up
    # --------------------------------------------------------------------------------------------------

    # Columns that do not seem important: PassengerID, Name, Ticket

    # Convert sex to 0/1
    combined['Sex'] = combined['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

    # Get Cabin letter from Cabin and convert to number
    combined['CabinLetter'] = combined.Cabin.str.extract('([A-Za-z])', expand=False)
    combined['CabinLetter'] = combined['CabinLetter'].fillna("NA")
    combined['CabinLetter'] = combined['CabinLetter'].map( {'D': 1, 'E': 2, 'B': 3, 'F': 4, 'C': 5, 'G': 6, 'A': 7, 'NA': 8, 'T': 9} ).astype(int)

    # Clean up NAs from fare
    combined['Fare'].fillna(combined['Fare'].dropna().mean(), inplace=True)

    # Clean up NAs from age
    guess_ages = np.zeros((2,3))
    guess_ages_test = np.zeros((2,3))
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = combined[(combined['Sex'] == i) & (combined['Pclass'] == j+1)]['Age'].dropna()

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

    for i in range(0, 2):
        for j in range(0, 3):
            combined.loc[(combined.Age.isnull()) & (combined.Sex == i) & (combined.Pclass == j+1), 'Age'] = guess_ages[i,j]

    combined['Age'] = combined['Age'].astype(int)

    # Convert Age into ranges
    combined['AgeRange'] = combined['Age']
    combined.loc[ combined['AgeRange'] <= 6, 'AgeRange'] = 0
    combined.loc[(combined['AgeRange'] > 6) & (combined['AgeRange'] <= 21), 'AgeRange'] = 1
    combined.loc[(combined['AgeRange'] > 21) & (combined['AgeRange'] <= 26), 'AgeRange'] = 2
    combined.loc[(combined['AgeRange'] > 26) & (combined['AgeRange'] <= 36), 'AgeRange'] = 3
    combined.loc[(combined['AgeRange'] > 36), 'AgeRange'] = 4

    # Get titles from name
    combined['Title'] = combined.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    combined['Title'] = combined['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
    	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    combined['Title'] = combined['Title'].replace('Mlle', 'Miss')
    combined['Title'] = combined['Title'].replace('Ms', 'Miss')
    combined['Title'] = combined['Title'].replace('Mme', 'Mrs')

    # Convert titles to numbers
    combined['Title'] = combined['Title'].fillna("Unk")
    combined['Title'] = combined['Title'].map( {'Unk': -1, 'Rare': 0, 'Miss': 1, 'Master': 2, 'Mr': 3, 'Mrs': 4} ).astype(int)

    # Convert Embarked to numbers
    combined['Embarked'] = combined['Embarked'].fillna("S")
    combined['Embarked'] = combined['Embarked'].map( {'C': 0, 'Q': 1, 'S': 2} ).astype(int)

    # Find number of family members on board
    combined['FamilyMems'] = combined['Parch'] + combined['SibSp']

    # Get fare ranges
    combined.loc[ combined['Fare'] <= 7.91, 'Fare'] = 0
    combined.loc[(combined['Fare'] > 7.91) & (combined['Fare'] <= 14.454), 'Fare'] = 1
    combined.loc[(combined['Fare'] > 14.454) & (combined['Fare'] <= 31), 'Fare']   = 2
    combined.loc[ combined['Fare'] > 31, 'Fare'] = 3
    combined['Fare'] = combined['Fare'].astype(int)

    # Create alone varibles
    combined['IsAlone'] = 0
    combined.loc[combined['FamilyMems'] == 1, 'IsAlone'] = 1

    # Variables from ahmedbesbes's solution
    # a function that extracts each prefix of the ticket, returns 'XXX' if no prefix (i.e the ticket is a digit)
    combined['Ticket'] = combined['Ticket'].map(cleanTicket)

    # introducing other features based on the family size
    combined['Singleton'] = combined['FamilyMems'].map(lambda s : 1 if s == 1 else 0)
    combined['SmallFamily'] = combined['FamilyMems'].map(lambda s : 1 if 2<=s<=4 else 0)
    combined['LargeFamily'] = combined['FamilyMems'].map(lambda s : 1 if 5<=s else 0)

    combined = combined[["Pclass", "Sex", "AgeRange", "Title", 'CabinLetter', 'Embarked', 'FamilyMems', 'Fare', 'IsAlone', "Survived", "Ticket", "Singleton", "SmallFamily", "LargeFamily"]]

    train = combined.ix[0:890]
    test = combined.ix[891:]

    return (train, test.drop(["Survived"], axis=1), ids)


def get_dummy_data():
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    ids = test['PassengerId']

    # Combine the two data sets just to make sure each end with same set of features
    combined = train.append(test)
    combined.reset_index(inplace=True)
    combined.drop('index',inplace=True,axis=1)

    # Data Clean Up
    # --------------------------------------------------------------------------------------------------

    # Columns that do not seem important: PassengerID, Name, Ticket

    # Convert sex to 0/1
    combined['Sex'] = combined['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

    # Get Cabin letter from Cabin and convert to number
    combined['Cabin'] = combined.Cabin.str.extract('([A-Za-z])', expand=False)
    combined['Cabin'] = combined['Cabin'].fillna("NA")
    combined = pd.get_dummies(combined, columns = ['Cabin'])

    # Clean up NAs from fare
    combined['Fare'].fillna(combined['Fare'].dropna().mean(), inplace=True)

    # Clean up NAs from age
    guess_ages = np.zeros((2,3))
    guess_ages_test = np.zeros((2,3))
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = combined[(combined['Sex'] == i) & (combined['Pclass'] == j+1)]['Age'].dropna()

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

    for i in range(0, 2):
        for j in range(0, 3):
            combined.loc[(combined.Age.isnull()) & (combined.Sex == i) & (combined.Pclass == j+1), 'Age'] = guess_ages[i,j]

    combined['Age'] = combined['Age'].astype(int)

    # Convert Age into ranges
    combined.loc[ combined['Age'] <= 6, 'Age'] = 0
    combined.loc[(combined['Age'] > 6) & (combined['Age'] <= 21), 'Age'] = 1
    combined.loc[(combined['Age'] > 21) & (combined['Age'] <= 26), 'Age'] = 2
    combined.loc[(combined['Age'] > 26) & (combined['Age'] <= 36), 'Age'] = 3
    combined.loc[(combined['Age'] > 36), 'Age'] = 4

    combined = pd.get_dummies(combined, columns = ['Age'])

    # Get titles from name
    combined['Title'] = combined.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    combined['Title'] = combined['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
    	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    combined['Title'] = combined['Title'].replace('Mlle', 'Miss')
    combined['Title'] = combined['Title'].replace('Ms', 'Miss')
    combined['Title'] = combined['Title'].replace('Mme', 'Mrs')

    # Convert titles to numbers
    combined['Title'] = combined['Title'].fillna("Unk")

    combined = pd.get_dummies(combined, columns = ['Title'])

    # Convert Embarked to numbers
    combined['Embarked'] = combined['Embarked'].fillna("S")
    combined = pd.get_dummies(combined, columns = ['Embarked'])

    # Find number of family members on board
    combined['FamilyMems'] = combined['Parch'] + combined['SibSp']

    # Get fare ranges
    combined.loc[ combined['Fare'] <= 7.91, 'Fare'] = 0
    combined.loc[(combined['Fare'] > 7.91) & (combined['Fare'] <= 14.454), 'Fare'] = 1
    combined.loc[(combined['Fare'] > 14.454) & (combined['Fare'] <= 31), 'Fare']   = 2
    combined.loc[ combined['Fare'] > 31, 'Fare'] = 3
    combined['Fare'] = combined['Fare'].astype(int)

    # Variables from ahmedbesbes's solution
    # a function that extracts each prefix of the ticket, returns 'XXX' if no prefix (i.e the ticket is a digit)
    combined['Ticket'] = combined['Ticket'].map(cleanTicket)

    # introducing other features based on the family size
    combined['Singleton'] = combined['FamilyMems'].map(lambda s : 1 if s == 1 else 0)
    combined['SmallFamily'] = combined['FamilyMems'].map(lambda s : 1 if 2<=s<=4 else 0)
    combined['LargeFamily'] = combined['FamilyMems'].map(lambda s : 1 if 5<=s else 0)


    combined = pd.get_dummies(combined, columns = ['FamilyMems'])
    combined = pd.get_dummies(combined, columns = ['Fare'])
    combined = pd.get_dummies(combined, columns = ['Pclass'])
    combined = pd.get_dummies(combined, columns = ['Ticket'])

    combined = combined.drop(["PassengerId", "Name", "SibSp", "Parch"], axis=1)

    train = combined.ix[0:890]
    test = combined.ix[891:]

    return (train, test.drop(["Survived"], axis=1), ids)
