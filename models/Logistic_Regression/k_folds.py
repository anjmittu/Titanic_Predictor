# data analysis and wrangling
import numpy as np
import data_cleanup as dc

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score

(train, test, ids) = dc.get_dummy_data()

num_test = 0.20
X_train = train.drop("Survived", axis=1)
Y_train = train["Survived"]

model = LogisticRegression()


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
