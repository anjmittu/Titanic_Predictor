# data analysis and wrangling
import pandas as pd
import data_cleanup as dc
import models.ml_helpers as mlh

# machine learning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

(train, test, ids) = dc.get_dummy_data()

num_test = 0.20
X_all = train.drop("Survived", axis=1)
Y_all = train["Survived"]
X_test  = test

# Pick a subset of parameters to use
clf = RandomForestClassifier()
clf = clf.fit(X_all, Y_all)

features = pd.DataFrame()
features['feature'] = X_all.columns
features['importance'] = clf.feature_importances_
features.sort_values(['importance'],ascending=False)

X_all_new = X_all[["Title_Mr", "Sex", "Title_Mrs", "Pclass_3", "Title_Miss", "Cabin_NA", "Fare_0", "Age_3", "Age_2", "Embarked_C"]]
X_test_new = X_test[["Title_Mr", "Sex", "Title_Mrs", "Pclass_3", "Title_Miss", "Cabin_NA", "Fare_0", "Age_3", "Age_2", "Embarked_C"]]

# model = SelectFromModel(clf, prefit=True)
# X_all_new = model.transform(X_all)
X_all_new.shape
# X_test_new = model.transform(X_test)
X_test_new.shape

# Choose some parameter combinations to try for grid search
parameters = {'n_estimators': [10, 100, 1000],
              'max_features': ['log2', 'sqrt'],
              'criterion': ['entropy', 'gini'],
              'max_depth': [3, 30, 100, 300, 1000],
              'min_samples_split': [1.0, 3, 10, 30],
              'min_samples_leaf': [1, 3, 10, 30]
             }

model = RandomForestClassifier(n_estimators=5, min_samples_leaf=3)
# model = mlh.grid_search(model, parameters, X_all_new, Y_all)

mlh.k_folds(model, X_all_new.as_matrix(), Y_all.as_matrix())

X_train, X_dev, Y_train, Y_dev = train_test_split(X_all_new, Y_all, test_size=num_test, random_state=23)

#Find score of model on training set and dev set
model.fit(X_train, Y_train)
acc_random_forest = round(model.score(X_train, Y_train) * 100, 2)
acc_random_forest
acc_random_forest = round(model.score(X_dev, Y_dev) * 100, 2)
acc_random_forest
#train model on all the data
model.fit(X_all_new, Y_all)
Y_pred = model.predict(X_test_new)

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': Y_pred.astype(int) })
output.to_csv('predictions/titanic_predictions_RF.csv', index = False)
