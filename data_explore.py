# data analysis and wrangling
import data_cleanup as dc

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

(train, test) = dc.get_data()

# Embarked could be useful
train[['Fare', 'Survived']].groupby(['Fare'], as_index=False).mean().sort_values(by='Survived', ascending=False)


grid = sns.FacetGrid(train, col='Survived', row='CabinLetter', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Fare', alpha=.5, bins=20)
grid.add_legend();


train.head()
