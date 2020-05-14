# -*- coding: utf-8 -*-py
import argparse
from datetime import datetime
from helpers import load_data
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from settings.settings import ROOT_PATH

ROOT_LOG_PATH = ROOT_PATH + 'final_exp_simplify/logs_grid_search/'

PARAM_GRID = {
    'clf__n_estimators': [50, 100, 200],
    'clf__max_depth': [None, 100],
    'clf__max_features': ['auto', 'log2'],
    'clf__criterion': ['gini', 'entropy'],
}

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--features_path", required=True, help="input folder to take the files from" +
                                                                "must be a train folder")
    args = vars(ap.parse_args())

    date_begin = datetime.now()
    X_train, y_train = load_data(args["features_path"])
    f = open(ROOT_LOG_PATH+args["features_path"].split('/')[1], "w+")

    # GridSearch CV process
    f.write("Process for GridSearchCV began at: {0} \n".format(date_begin))
    pipe_log = Pipeline([('min_max_scaler', preprocessing.MinMaxScaler()),
                        ('clf', RandomForestClassifier())]) 
    log_grid = GridSearchCV(pipe_log, param_grid=PARAM_GRID, scoring="accuracy",
                            verbose=3, cv=5, n_jobs=-1)

    fitted = log_grid.fit(X_train, y_train)
    date_end_cv = datetime.now()
    f.write("Gridsearch CV ended at : {0}, took : {1} \n".format(date_end_cv, date_end_cv - date_begin))
    best_params, best_score = log_grid.best_params_, log_grid.best_score_
    f.write("Best params: {0} \n".format(best_params))
    f.write("Best score: {0} \n".format(best_score))