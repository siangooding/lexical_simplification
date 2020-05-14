# -*- coding: utf-8 -*-
import pickle
import argparse
from sklearn.metrics import classification_report
from settings.settings import ROOT_PATH

RES_PATH = ROOT_PATH + 'final_exp_simplify/results/{0}'

def build_y_matrix(res_path):
    # matrix: col = actual, line = predicted
    res = pickle.load(open(res_path, 'rb'))
    matrix = [[0]*3]*3
    y_true, y_pred = [], []

    for level, files_info in res.items():
        for _, file_info in files_info.items():
            y_true.append(int(level))
            y_pred.append(int(file_info['level']))
            matrix[int(file_info['level'])-1][int(level)-1] += 1
    
    return matrix, y_true, y_pred


def analyse_results(res_path):
    matrix, y_true, y_pred = build_y_matrix(res_path)
    print('Printing confusion matrix')
    for l in matrix:
        print(l)
    print(classification_report(y_true, y_pred, labels=[1, 2, 3])) 


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="file name containing the results" +
                                                         "should be in the final_exp_simplify/results/ folder")
    args = vars(ap.parse_args())

    analyse_results(res_path=RES_PATH.format(args["input"]))