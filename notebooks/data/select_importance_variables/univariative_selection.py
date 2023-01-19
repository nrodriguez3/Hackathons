import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# Read in data and display first 5 rows
features = pd.read_csv('temps.csv')
# One-hot encode the data using pandas get_dummies
features = pd.get_dummies(features)
# Display the first 5 rows of the last 12 columns
features.iloc[:,5:].head(5)

# Labels are the values we want to predict
labels = np.array(features['actual'])
# Remove the labels from the features
# axis 1 refers to the columns
features= features.drop('actual', axis = 1)
# Saving feature names for later use
feature_list = list(features.columns)
# Convert to numpy array

bestfeatures = SelectKBest(score_func=chi2, k=4)
fit = bestfeatures.fit(features, labels)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(feature_list)
#concat two dataframes for better visualization 

featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(2,'Score'))  #print 10 best features