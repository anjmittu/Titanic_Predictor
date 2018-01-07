# data analysis and wrangling
import pandas as pd
import data_cleanup as dc

# machine learning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

(train, test) = dc.get_data()

num_test = 0.20
X_train = train[["Pclass", "Sex", "AgeRange", "Title", 'CabinLetter', 'Embarked', 'FamilyMems', 'Fare']]
Y_train = train["Survived"]
X_train, X_CV, Y_train, Y_CV = train_test_split(X_train, Y_train, test_size=num_test, random_state=23)
X_test  = test[["Pclass", "Sex", "AgeRange", "Title", 'CabinLetter', 'Embarked', 'FamilyMems', 'Fare']]
ids = test['PassengerId']

# Using Random Forest
random_forest = RandomForestClassifier(n_estimators=10, criterion="gini", max_depth=30, max_features='log2', min_samples_leaf=3, min_samples_split=3)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest
acc_random_forest = round(random_forest.score(X_CV, Y_CV) * 100, 2)
acc_random_forest

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': Y_pred })
output.to_csv('titanic-predictions.csv', index = False)
