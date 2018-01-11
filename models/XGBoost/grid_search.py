# data analysis and wrangling
import data_cleanup as dc

# machine learning
from xgboost import XGBClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV

(train, test, ids) = dc.get_dummy_data()

num_test = 0.20
X_train = train.drop("Survived", axis=1)
Y_train = train["Survived"]
X_test  = test

# Choose the type of classifier.
clf = XGBClassifier()

# Choose some parameter combinations to try
parameters = {'max_depth': [3, 10, 30, 100, 300],
              'learning_rate': [.01, .1, .3, 1, 3],
              'n_estimators': [10, 30, 100, 300, 1000],
              'booster': ["gbtree", "gblinear", "dart"],
              'gamma': [0, .1, .3, 1],
              'min_child_weight': [1, 3, 10, 30]
             }

# Type of scoring used to compare parameter combinations
acc_scorer = make_scorer(accuracy_score)

# Run the grid search
grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)
grid_obj = grid_obj.fit(X_train, Y_train)

print(grid_obj.best_score_)
print(grid_obj.best_params_)
