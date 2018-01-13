# data analysis and wrangling
import numpy as np
import data_cleanup as dc

# machine learning
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import GridSearchCV

def k_folds(model, X_train, Y_train):
    num_test = 0.20

    kf = KFold(891, n_folds=10)
    outcomes = []
    fold = 0
    for train_index, test_index in kf:
        fold += 1
        Xf_train, Xf_test = X_train.values[train_index], X_train.values[test_index]
        Yf_train, Yf_test = Y_train.values[train_index], Y_train.values[test_index]
        model.fit(Xf_train, Yf_train)
        predictions = model.predict(Xf_test)
        accuracy = accuracy_score(Yf_test, predictions)
        outcomes.append(accuracy)
        print("Fold {0} accuracy: {1}".format(fold, accuracy))

    mean_outcome = np.mean(outcomes)
    print("Mean Accuracy: {0}".format(mean_outcome))

def k_folds_array(model, X_train, Y_train):
    num_test = 0.20

    kf = KFold(891, n_folds=10)
    outcomes = []
    fold = 0
    for train_index, test_index in kf:
        fold += 1
        Xf_train, Xf_test = X_train[train_index], X_train[test_index]
        Yf_train, Yf_test = Y_train[train_index], Y_train[test_index]
        model.fit(Xf_train, Yf_train)
        predictions = model.predict(Xf_test)
        accuracy = accuracy_score(Yf_test, predictions)
        outcomes.append(accuracy)
        print("Fold {0} accuracy: {1}".format(fold, accuracy))

    mean_outcome = np.mean(outcomes)
    print("Mean Accuracy: {0}".format(mean_outcome))

def grid_search(model, parameters, X_train, Y_train):
    # Type of scoring used to compare parameter combinations
    acc_scorer = make_scorer(accuracy_score)

    # Run the grid search
    grid_obj = GridSearchCV(model, parameters, scoring=acc_scorer)
    grid_obj = grid_obj.fit(X_train, Y_train)

    print(grid_obj.best_params_)

    # Set the clf to the best combination of parameters
    return grid_obj.best_estimator_
