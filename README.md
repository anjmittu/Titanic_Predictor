# Titanic_Predictor
Kaggle's "Titanic: Machine Learning from Disaster" Competition

This is my try at competition.  Currently I am ranked 1208 of 9553.  My best score is 79.904

## My Approach
### EDA
I tried working using all categorical features and all dummy features.  This was done in the file **data_cleanup.py**. The steps I took to prepare both sets of features is below:

1. I immediately got rid of the _PassengerID_ as this would be no help.
2. I changed sex from "male",  and "female" to 0 and 1.  I will do a lot of changes from string to ints because most of the ml models work best with ints
3. I extracted the first letter from the _Cabin_ feature, converted it to an int and set it as the _CabinLetter_.  This is done because alone _Cabin_ does not tell us much.  Most of the passengers have a unique cabin number, however just the letter from the cabin gives a bit more information
3. I replaced the missing _Age_ values by guessing the persons age as the median age for their _Sex_ and _Pclass_.  _Age_ was then converted into five different _AgeRange_.  The values for the ranges were chosen by the distribution of ages.
4. I created the feature _Title_ by extracted each passenger's title from their _Name_.  _Name_ alone does not give much information since each full name is different.  However, _Title_ is useful since it also can indicate what class the person is in.
5. I filled the missing values in _Embarked_ with the most common value "S".  I then converted the values to ints.
6. I created the feature _FamilyMems_ by combining _Parch_ and _SibSp_.  This gives the total number of family members a passenger has onboard.
7. I filled the missing values in _Fare_ with the most common value "NA".  _Fare_ was then converted into five different ranges.  The values for the ranges were chosen
by the distribution of fares.
8. I extracted the prefix of the each ticket and used 'XXX' for ones without a prefix.
9. I added three features _Singleton_, _SmallFamily_, and _LargeFamily_ based on _FamilyMems_.


### Models
I tried four different models: Logistical Regression, Random Forest, Adaboost, and XGBoost.  Random Forest, Adaboost, and XGBoost all preformed fairly similarly so I decided to focus on Random Forest.  My models seem to be overfitting the data (exact values from all runs are in **notes.xlsx**).  The scores on the training sets are much higher than the test sets.  Because of this I have decreased the number of features used, however this doesn't seem to be helping. I might want to decrease even more.

I've decreased the amount of features to just the top 10 out of ~70.  This has been preforming better. Now I'll try with different amount of features.

All of the models are described below.

##### Logistical Regression
Best score: 76.555 with features: "Pclass, "Sex", "AgeRange", "Title"

#### Random Forest
Best score: 79.904 with features: Title_Mr, "Sex", "Title_Mrs", "Pclass_3", "Title_Miss", "Cabin_NA", "Fare_0", "Age_3", "Age_2", "Embarked_C" and default hyperparameters

Previous best score: 77.99 with features: Pclass, "Sex", "AgeRange", "Title", 'CabinLetter', 'Embarked', 'FamilyMems', 'Fare', 'IsAlone' and hyperparameters: {'criterion': 'entropy', 'max_depth': 30, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 30, 'n_estimators': 10}

#### Adaboost
Best score: 77.99 with features: Pclass, "Sex", "AgeRange", "Title", 'CabinLetter', 'Embarked', 'FamilyMems', 'Fare' and hyperparameters: {n_estimators=10}

#### XGBoost
Best score: 77.511 with features: With Dummy Vars and hyperparameters: {'booster': 'gbtree', 'gamma': 0, 'learning_rate': 0.3, 'max_depth': 3, 'min_child_weight': 10, 'n_estimators': 30}
