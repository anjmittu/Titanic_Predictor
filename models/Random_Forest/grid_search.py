# data analysis and wrangling
import data_cleanup as dc

# machine learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV

(train, test, ids) = dc.get_dummy_data()

num_test = 0.20
X_train = train.drop("Survived", axis=1)
Y_train = train["Survived"]
X_test  = test

# Choose the type of classifier.
clf = RandomForestClassifier()

# Choose some parameter combinations to try
parameters = {'n_estimators': [10, 100, 1000],
              'max_features': ['log2', 'sqrt', 'auto'],
              'criterion': ['entropy', 'gini'],
              'max_depth': [3, 30, 100, 300, 1000],
              'min_samples_split': [1.0, 3, 10, 30],
              'min_samples_leaf': [1, 3, 10, 30]
             }

# Type of scoring used to compare parameter combinations
acc_scorer = make_scorer(accuracy_score)

# Run the grid search
grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)
grid_obj = grid_obj.fit(X_train, Y_train)

print(grid_obj.best_score_)
print(grid_obj.best_params_)

# Set the clf to the best combination of parameters
clf = grid_obj.best_estimator_

# Fit the best algorithm to the data.
clf.fit(X_train, Y_train)
acc_random_forest = round(clf.score(X_train, Y_train) * 100, 2)
print(acc_random_forest)
