# data analysis and wrangling
import pandas as pd
import data_cleanup as dc

# machine learning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

(train, test) = dc.get_data()

num_test = 0.20
X_all = train[["Pclass", "Sex", "AgeRange", "Title", 'CabinLetter', 'Embarked', 'FamilyMems', 'Fare']]
Y_all = train["Survived"]
X_train, X_dev, Y_train, Y_dev = train_test_split(X_all, Y_all, test_size=num_test, random_state=23)
X_test  = test[["Pclass", "Sex", "AgeRange", "Title", 'CabinLetter', 'Embarked', 'FamilyMems', 'Fare']]
ids = test['PassengerId']

# Using Random Forest
model = RandomForestClassifier(n_estimators=10, criterion="entropy", max_depth=None, max_features='log2', min_samples_leaf=1, min_samples_split=5)
#Find score of model on training set and dev set
model.fit(X_train, Y_train)
acc_random_forest = round(model.score(X_train, Y_train) * 100, 2)
acc_random_forest
acc_random_forest = round(model.score(X_dev, Y_dev) * 100, 2)
acc_random_forest
#train model on all the data
model.fit(X_all, Y_all)
Y_pred = model.predict(X_test)

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': Y_pred })
output.to_csv('titanic_predictions_RF.csv', index = False)
