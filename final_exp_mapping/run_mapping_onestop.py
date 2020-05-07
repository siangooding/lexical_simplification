# -*- coding: utf-8 -*-
import pickle
import argparse
from tqdm import tqdm
from os import listdir
from settings.settings import ROOT_PATH
from pipeline_readability import ReadabilityPipeline

REL_PATH_TO_KEY = {
    'OneStopEnglishCorpus/adv-text/': 'adv',
    'OneStopEnglishCorpus/int-text/': 'int',
    'OneStopEnglishCorpus/ele-text/': 'ele'
}
STORING = 'final_exp_mapping/mappings/mapping_one_stop_{0}.pkl'

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--features", required=True, help="features to consider, separated by `_`" +
                                                            "currently implemented: T_L_SGR_NER")
    args = vars(ap.parse_args())

    res_pickle = {}
    pipeline = ReadabilityPipeline()
    for rel_path, pointer in REL_PATH_TO_KEY.items():
        res_pickle[pointer] = {}
        files = listdir(ROOT_PATH + rel_path)
        for f in tqdm(files):
            level, timestamps = \
                pipeline.process_pipeline(text_path=ROOT_PATH + rel_path + f,
                                          folder_xml=ROOT_PATH+'OneStopEnglishCorpus/{0}-text-xml/'.format(pointer),
                                          features=args["features"],
                                          train_outputfile="mapping-feats", simplified_path=None)
            res_pickle[pointer][f] = {'level': level, 'timestamps': timestamps}
    
    with open(ROOT_PATH+STORING.format(args["features"]), 'wb') as outfile:
        pickle.dump(res_pickle, outfile)
