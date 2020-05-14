# -*- coding: utf-8 -*-

import os
import argparse
import pickle
from os import listdir
from pipeline.pipeline_readability import ReadabilityPipeline
from settings.settings import ROOT_PATH

RECURSIVE_CONFIG_PATH = ROOT_PATH + 'final_exp_simplify/recursive_config_exp.yaml'
READABILITY_CONFIG_PATH = ROOT_PATH + 'readability_final_exp_simplifyode/readability_config_exp.yaml'
STORING = ROOT_PATH + 'results/{0}'

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--save_path", required=True, help="results save file name. Must be a .pkl file")
    ap.add_argument("-t", "--test_path", required=True, herelp="full folder path containing test files by folder levels")
    args = vars(ap.parse_args())

    readability_pipeline = ReadabilityPipeline(include_simplification=False,
                                               readability_config_path=READABILITY_CONFIG_PATH)
    
    res = {}
    for level in listdir(args["test_path"]):
        res[level] = {}
        for file_name in listdir(os.path.join(args["test_path"], level)):
            level, timestamps = \
                readability_pipeline.process_pipeline(text_path=os.path.join(args["test_path"], level, file_name))
            res[level][file_name] = {'level': level, 'timestamps': timestamps}
    
    with open(STORING.format(args["save_path"]), 'wb') as outfile:
        pickle.dump(res, outfile)
