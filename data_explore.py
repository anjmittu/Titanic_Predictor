# data analysis and wrangling
import pandas as pd
import data_cleanup as dc

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

raw_train = pd.read_csv('data/train.csv')
raw_test = pd.read_csv('data/test.csv')

(train, test, ind) = dc.get_data()

# Embarked could be useful
raw_train[['Cabin', 'Survived']].groupby(['Cabin'], as_index=False).mean().sort_values(by='Survived', ascending=False)

train[['CabinLetter', 'Survived']].groupby(['CabinLetter'], as_index=False).mean().sort_values(by='Survived', ascending=False)


grid = sns.FacetGrid(train, col='Survived', row='CabinLetter', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Fare', alpha=.5, bins=20)
grid.add_legend();


train.head()
