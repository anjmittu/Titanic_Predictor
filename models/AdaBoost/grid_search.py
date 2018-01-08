# data analysis and wrangling
import data_cleanup as dc

# machine learning
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV

(train, test) = dc.get_data()

num_test = 0.20
X_train = train[["Pclass", "Sex", "AgeRange", "Title", 'CabinLetter', 'Embarked', 'FamilyMems', 'Fare']]
Y_train = train["Survived"]
X_test  = test[["Pclass", "Sex", "AgeRange", "Title", 'CabinLetter', 'Embarked', 'FamilyMems', 'Fare']]
ids = test['PassengerId']

# Choose the type of classifier.
clf = AdaBoostClassifier()

# Choose some parameter combinations to try
parameters = {'n_estimators': [10, 50, 100, 1000],
              'learning_rate': [.001, .1, 1, 3, 10]
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
