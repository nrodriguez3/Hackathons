import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
import pydot
from sklearn.tree import export_graphviz

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

# Split the data into training and testing sets
rf, mape, accuracy = pipeline_model(features, labels)
feature_importance = get_feature_importance(rf)
important_indices = [feature for feature, importance in feature_importance if importance > 0.3]
print(important_indices)

rf_with_fi, mape, accuracy = retrain_rf(features, important_indices)

def pipeline_model(features, labels):
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)
    # Instantiate model with 1000 decision trees
    rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
    # Train the model on training data
    rf.fit(train_features, train_labels)
    # Use the forest's predict method on the test data
    predictions = rf.predict(test_features)
    mape, accuracy  = get_metrics(predictions, test_labels)

    return rf, mape, accuracy

def get_metrics(predictions, test_labels):
    # Calculate the absolute errors
    errors = abs(predictions - test_labels)
    mape = 100 * (errors / test_labels)
    # Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    return mape, accuracy


def plot_forest(rf, path = 'tree.png'):
    # Pull out one tree from the forest
    tree = rf.estimators_[5]
    # Export the image to a dot file
    export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)
    # Use dot file to create a graph
    (graph, ) = pydot.graph_from_dot_file('tree.dot')
    # Write graph to a png file
    graph.write_png(f'{path}')
    return None

def get_feature_importance(rf):
    # Get numerical feature importances
    importances = list(rf.feature_importances_)
    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
    return feature_importances


def retrain_rf(features, important_indices):
    rf_most_important = RandomForestRegressor(n_estimators= 1000, random_state=42)
    # Extract the two most important features
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)
    train_important = train_features[:, important_indices]
    test_important = test_features[:, important_indices]
    # Train the random forest
    rf_most_important.fit(train_important, train_labels)
    # Make predictions and determine the error
    predictions = rf_most_important.predict(test_important)
    mape, accuracy  = get_metrics(predictions, test_labels)

    return rf_most_important, mape, accuracy

