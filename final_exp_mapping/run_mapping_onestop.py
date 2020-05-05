# -*- coding: utf-8 -*-
import pickle
import argparse
from os import listdir
from settings.settings import ROOT_PATH
from ..pipeline_readability import ReadabilityPipeline

REL_PATH_TO_KEY = {
    'onestop/adv-text/': 'adv',
    'onestop/mod-text/': 'mod',
    'onestop/ele-text/': 'ele'
}
STORING = 'final_exp_mapping/mapping_one_stop_{0}.pkl'

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--features", required=True, help="features to consider, separated by `_`" +
                                                            "currently implemented: T_L_SGR_NER")
    args = vars(ap.parse_args())

    res_pickle = {}
    pipeline = ReadabilityPipeline()
    for rel_path, pointer in REL_PATH_TO_KEY.items():
        res_pickle[pointer] = {}
        files = [ROOT_PATH + rel_path + file for file in listdir(ROOT_PATH + rel_path)]
        for f in files:
            level, timestamps = \
                ReadabilityPipeline.process_pipeline(text_path=f,
                                                     folder_xml=ROOT_PATH+'onestop/{0}-text-xml/'.format(pointer),
                                                     features=args["features"],
                                                     train_outputfile="mapping-feats", simplified_path=None)
            res_pickle[pointer][f] = {'level': level, 'timestamps': timestamps}
    
    with open(ROOT_PATH+STORING.format(args["features"]), 'wb') as outfile:
        pickle.dump(res_pickle, outfile)