# data analysis and wrangling
import pandas as pd
import data_cleanup as dc

# machine learning
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

(train, test, ids) = dc.get_dummy_data()

num_test = 0.20
X_all = train.drop("Survived", axis=1)
Y_all = train["Survived"]
X_train, X_dev, Y_train, Y_dev = train_test_split(X_all, Y_all, test_size=num_test, random_state=23)
X_test  = test

# Starting with a simple log regresssion
model = XGBClassifier(booster='gbtree', gamma=0, learning_rate=.3, max_depth=3, min_child_weight=10, n_estimators=30)
model.fit(X_train, Y_train)
acc_random_forest = round(model.score(X_train, Y_train) * 100, 2)
acc_random_forest
acc_random_forest = round(model.score(X_dev, Y_dev) * 100, 2)
acc_random_forest
#train model on all the data
model.fit(X_all, Y_all)
Y_pred = model.predict(X_test[list(X_all.columns.values)])

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': Y_pred })
output.to_csv('titanic-predictions_XGB.csv', index = False)
