# data analysis and wrangling
import pandas as pd
import data_cleanup as dc

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

(train, test) = dc.get_data()

num_test = 0.20
X_train = train[["Pclass", "Sex", "AgeRange", "Title", 'CabinLetter', 'Embarked', 'FamilyMems', 'Fare']]
Y_train = train["Survived"]
X_train, X_CV, Y_train, Y_CV = train_test_split(X_train, Y_train, test_size=num_test, random_state=23)
X_test  = test[["Pclass", "Sex", "AgeRange", "Title", 'CabinLetter', 'Embarked', 'FamilyMems', 'Fare']]
ids = test['PassengerId']

# Starting with a simple log regresssion
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log

coeff_df = pd.DataFrame(X_train.columns)
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': Y_pred })
output.to_csv('titanic-predictions.csv', index = False)
