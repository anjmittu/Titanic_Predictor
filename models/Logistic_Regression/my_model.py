# data analysis and wrangling
import pandas as pd
import data_cleanup as dc

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

(train, test, ids) = dc.get_dummy_data()

num_test = 0.20
X_all = train.drop("Survived", axis=1)
Y_all = train["Survived"]
X_train, X_dev, Y_train, Y_dev = train_test_split(X_all, Y_all, test_size=num_test, random_state=23)
X_test  = test

# Starting with a simple log regresssion
model = LogisticRegression()
model.fit(X_train, Y_train)
acc_random_forest = round(model.score(X_train, Y_train) * 100, 2)
acc_random_forest
acc_random_forest = round(model.score(X_dev, Y_dev) * 100, 2)
acc_random_forest
#train model on all the data
model.fit(X_all, Y_all)
Y_pred = model.predict(X_test)

coeff_df = pd.DataFrame(X_train.columns)
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': Y_pred })
output.to_csv('titanic-predictions.csv', index = False)
