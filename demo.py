import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import tree

if __name__ == '__main__':
    train = pd.read_csv('input/train.csv')
    test = pd.read_csv('input/test.csv')
    # print(titanic.head())
    # print(titanic.describe())

    d = train[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch']]
    print(d.isnull().sum())
    print(d.groupby(['Sex', 'Pclass']).Age.apply(lambda x: x.isnull().sum()) / d.groupby(['Sex', 'Pclass']).Age.count())

    print(d['Age'].mean())
    d['Sex'] = np.where(d.Sex == 'female', 1, 0)
    d['Age'] = d['Age'].fillna(d['Age'].mean())

    print(d.Survived.value_counts())
    print(d.Survived.mean())
    print(d.groupby('Sex').Survived.mean())
    print(d.groupby('Pclass').Survived.mean())
    print(d.groupby(['Sex', 'Pclass']).Survived.mean())

    d['Spouse'] = ((d.Age > 18) & (d.SibSp >= 1)).astype(int)
    print(d.Spouse.value_counts())
    print(d.groupby(['Pclass', 'Spouse']).Survived.mean())

    train = d
    test = pd.DataFrame(data=test, columns=d.columns)
    test['Sex'] = np.where(test.Sex == 'female', 1, 0)
    test['Age'] = test['Age'].fillna(d['Age'].mean())
    test['Spouse'] = ((test.Age > 18) & (test.SibSp >= 1)).astype(int)

    ctree = tree.DecisionTreeClassifier(random_state=1, max_depth=2)
    feature_df = train.drop('Survived', axis=1)
    survived = train['Survived']
    ctree.fit(feature_df, survived)

    print(ctree.classes_)

    # Create a feature vector
    features = d.columns.tolist()[1:]

    # Predict what will happen for 1st class woman
    print(features)
    ctree.predict_proba([1, 1, 25, 0, 0, 0])
    ctree.predict([1, 1, 25, 0, 0, 0])

    # Predict what will happen for a 3rd class man
    ctree.predict_proba([3, 0, 25, 0, 0, 0])

    # How about a woman?
    ctree.predict_proba([3, 1, 25, 0, 0, 0])

    # Which features are the most important?
    print(ctree.feature_importances_)

    # Clean up the output
    pd.DataFrame(zip(features, ctree.feature_importances_)).sort_index(by=1, ascending=False)

    # Make predictions on the test set
    preds = ctree.predict(test.drop('Survived', axis=1))

    # Calculate accuracy
    metrics.accuracy_score(test['Survived'], preds)

    # Confusion matrix
    pd.crosstab(test['Survived'], preds, rownames=['actual'], colnames=['predicted'])

    # Make predictions on the test set using predict_proba
    probs = ctree.predict_proba(test.drop('Survived', axis=1))[:, 1]

    # Calculate the AUC metric
    metrics.roc_auc_score(test['Survived'], probs)

    '''
    FINE-TUNING THE TREE
    '''

    from sklearn.cross_validation import cross_val_score
    from sklearn.grid_search import GridSearchCV

    y = d['Survived'].values
    X = d[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Spouse']].values

    # check CV score for max depth = 3
    ctree = tree.DecisionTreeClassifier(max_depth=3)
    np.mean(cross_val_score(ctree, X, y, cv=5, scoring='roc_auc'))

    # check CV score for max depth = 10
    ctree = tree.DecisionTreeClassifier(max_depth=10)
    np.mean(cross_val_score(ctree, X, y, cv=5, scoring='roc_auc'))

    # Conduct a grid search for the best tree depth
    ctree = tree.DecisionTreeClassifier(random_state=1, min_samples_leaf=20)
    depth_range = range(1, 20)
    param_grid = dict(max_depth=depth_range)
    grid = GridSearchCV(ctree, param_grid, cv=5, scoring='roc_auc')
    grid.fit(X, y)

    # Check out the scores of the grid search
    grid_mean_scores = [result[1] for result in grid.grid_scores_]

    print(grid_mean_scores)

    # Plot the results of the grid search
    plt.figure()
    plt.plot(depth_range, grid_mean_scores)
    plt.hold(True)
    plt.grid(True)
    plt.plot(grid.best_params_['max_depth'], grid.best_score_, 'ro', markersize=12, markeredgewidth=1.5,
             markerfacecolor='None', markeredgecolor='r')

    # Get the best estimator
    best = grid.best_estimator_

    print(best)

    # Read in test data from site
    test = pd.read_csv('test.csv')

    # Do all of the same transformations we did above to this set
    test['Sex'] = np.where(test.Sex == 'female', 1, 0)
    test['Age'] = test['Age'].fillna(test['Age'].mean())
    test['Spouse'] = ((test.Age > 18) & (test.SibSp >= 1)).astype(int)

    # predict our out of sample data using our "Best" model
    predictions = best.predict(test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Spouse']].values)

    submission_df = pd.DataFrame(test['PassengerId'])
    submission_df['Survived'] = predictions

    # Make our submission
    submission_df.to_csv('submission.csv', index=False)
