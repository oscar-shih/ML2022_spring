'''Reference code from https://www.kaggle.com/lemontreeyc/hw1-public-strong-baseline and do some modifications'''
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

data = pd.read_csv('/home/oscarshih/Desktop/ML/Hw1-Regression/covid.train.csv')

x = data[data.columns[1:117]]
y = data[data.columns[117]]

x = (x - x.min()) / (x.max() - x.min())

bestfeatures = SelectKBest(score_func=f_regression, k=10)
fit = bestfeatures.fit(x,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(x.columns)

featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['Specs', 'Score']  #name the data_frame columns
print(featureScores.nlargest(25, 'Score'))  #print 30 best features from train.csv