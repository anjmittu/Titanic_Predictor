# data analysis and wrangling
import pandas as pd
import data_cleanup as dc
import models.ml_helpers as mlh

# machine learning
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel

(train, test, ids) = dc.get_dummy_data()

num_test = 0.20
X_all = train.drop("Survived", axis=1)
Y_all = train["Survived"]
X_test  = test

# Pick a subset of parameters to use
clf = XGBClassifier()
clf = clf.fit(X_all, Y_all)
model = SelectFromModel(clf, prefit=True)
X_all_new = model.transform(X_all)
X_all_new.shape
X_test_new = model.transform(X_test)
X_test_new.shape

# Choose some parameter combinations to try for grid search
parameters = {'max_depth': [3, 10, 30, 100, 300],
              'learning_rate': [.01, .1, .3, 1, 3],
              'n_estimators': [10, 30, 100, 300, 1000],
              'booster': ["gbtree", "gblinear", "dart"],
              'gamma': [0, .1, .3, 1],
              'min_child_weight': [1, 3, 10, 30]
             }

model = XGBClassifier()
model = mlh.grid_search(model, parameters, X_all_new, Y_all)

mlh.k_folds(model, X_all_new, Y_all)

X_train, X_dev, Y_train, Y_dev = train_test_split(X_all_new, Y_all, test_size=num_test, random_state=23)

model.fit(X_train, Y_train)
acc_random_forest = round(model.score(X_train, Y_train) * 100, 2)
acc_random_forest
acc_random_forest = round(model.score(X_dev, Y_dev) * 100, 2)
acc_random_forest
#train model on all the data
model.fit(X_all, Y_all)
Y_pred = model.predict(X_test[list(X_all.columns.values)])

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': Y_pred })
output.to_csv('predictions/titanic-predictions_XGB.csv', index = False)
