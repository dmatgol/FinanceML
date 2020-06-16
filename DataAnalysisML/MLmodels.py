from sklearn import svm, neighbors, model_selection
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from DataAnalysisTECH import extract_featureset
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
import statsmodels.formula.api as smf
from sklearn import metrics
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix
import numpy as np


def buy_sell_hold_observations(y):
    """Function return the total number of observations in each group
    (1-buy), (0-hold), (-1,sell)"""
    return (np.around((np.array(y) == 1).sum()/len(y)*100, 3),
            np.around((np.array(y) == -1).sum()/len(y)*100, 3),
            np.around((np.array(y) == 0).sum()/len(y)*100, 3))


def prior_buysellhold_error_rate(confusion_matrix):
    """Prior error rate: 1- actual probability for each class
    Row 0 for -1(sell), Row 1 for 0 (hold), Row 2 for 1 (buy)"""
    return (1 - np.sum(confusion_matrix[2, :])/np.sum(confusion_matrix),
            1 - np.sum(confusion_matrix[1, :])/np.sum(confusion_matrix),
            1 - np.sum(confusion_matrix[0, :])/np.sum(confusion_matrix)
            )


def total_error_rate(confusion_matrix):
    return (1-np.sum(confusion_matrix.diagonal())/np.sum(confusion_matrix))


def recall(confusion_matrix):
    """Recal computation for each class:
    (1-buy), (0-hold), (-1,sell)
    Row 0 for -1(sell), Row 1 for 0 (hold), Row 2 for 1 (buy)"""
    return (confusion_matrix[0, 0]/np.sum(confusion_matrix[0, :]),
            confusion_matrix[1, 1]/np.sum(confusion_matrix[1, :]),
            confusion_matrix[2, 2]/np.sum(confusion_matrix[2, :])
            )


def precision(confusion_matrix):
    """Precision computation for each class:
    (1-buy), (0-hold), (-1,sell)
    Row 0 for -1(sell), Row 1 for 0 (hold), Row 2 for 1 (buy)"""
    return (confusion_matrix[0, 0]/np.sum(confusion_matrix[:, 0]),
            confusion_matrix[1, 1]/np.sum(confusion_matrix[:, 1]),
            confusion_matrix[2, 2]/np.sum(confusion_matrix[:, 2])
            )


def model_stats(confusion_matrix, y):
    return pd.Series({'Buy Observations': buy_sell_hold_observations(y)[0],
                      'Sell Observations': buy_sell_hold_observations(y)[1],
                      'Hold Observations': buy_sell_hold_observations(y)[2],
                      'Prior Buy error rate': prior_buysellhold_error_rate(confusion_matrix)[0],
                      'Prior Sell error rate': prior_buysellhold_error_rate(confusion_matrix)[2],
                      'Prior Hold error rate': prior_buysellhold_error_rate(confusion_matrix)[1],
                      'Total error rate': total_error_rate(confusion_matrix),
                      'Recall Buy': recall(confusion_matrix)[2],
                      'Recall Sell': recall(confusion_matrix)[0],
                      'Recall Hold': recall(confusion_matrix)[1],
                      'Precision Buy': precision(confusion_matrix)[2],
                      'Precision Sell': precision(confusion_matrix)[0],
                      'Precision Hold': precision(confusion_matrix)[1]
                      })


def distinct_stocks(previouslist, stocklist):
    s = set(previouslist)
    for stock in stocklist:
        s.add(str(stock))
    return s


def model_split(X, y, formula, k):
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=1,
                                                                        test_size=0.3)

    X_folds = np.array_split(X_train, k)
    y_folds = np.array_split(y_train, k)
    predictors = X.columns
    response = y.columns

    if formula == '':
        formula = '{} ~ {}'.format(response[0], '+'.join([p for p in predictors]))

    return (X_folds, y_folds, formula, X_train, X_test, y_train, y_test)


def do_linear_ml(ticker, formula='', k=10):
    """Perform k-fold cross validation to return mean MSE score"""

    X, y, df = extract_featureset(ticker)
    X_folds, y_folds, formula, X_train, X_test, y_train, y_test = model_split(X, y, formula, k)

    train_cv_rmse = []
    test_cv_rmse = []
    relevant_stocks = []
    for i in np.arange(len(X_folds)):
        X_test_cv = X_folds[i]
        y_test_cv = y_folds[i]
        X_train_cv = X_train.drop(X_test_cv.index)
        y_train_cv = y_train.drop(X_test_cv.index)
        model = smf.ols(formula, pd.concat([X_train_cv, y_train_cv], axis=1)).fit()
        prediction = model.predict(X_test_cv)
        prediction_test = model.predict(X_test)
        test_cv_rmse += [np.sqrt(metrics.mean_squared_error(y_test, prediction_test))]
        train_cv_rmse += [np.sqrt(metrics.mean_squared_error(y_test_cv, prediction))]
        relevant_stocks = distinct_stocks(relevant_stocks, model.pvalues[model.pvalues < 0.05].index.values)

    print(pd.DataFrame({'Relevant Stocks': list(relevant_stocks)}))
    print("Train RMSE: {}".format(np.mean(train_cv_rmse)))
    print("Test RMSE: {}".format(np.mean(test_cv_rmse)))


def train_mlmodel(ticker, mlmodel):
    """Receives different machine learning models and fits the training data
    to the model and outputs the train/test RMSE"""
    X, y, df = extract_featureset(ticker)
    _, _, _, X_train, X_test, y_train, y_test = model_split(X, y, '', 10)
    model = mlmodel.fit(X_train, y_train.values.ravel())
    y_pred = model.predict(X_test)
#   plot_confusion_matrix(model, X_test, y_test)

    print(model_stats(confusion_matrix(y_test, y_pred), y))
    print(classification_report(y_test, y_pred))
    print('Accuracy ', model.score(X_test, y_test))
    print('Predicted spread', Counter(y_pred))
    return y_pred


train_mlmodel("AAPL", VotingClassifier([('lsvc', svm.LinearSVC()),
                                        ('knn', KNeighborsClassifier()),
                                        ('rfor', RandomForestClassifier())]))
