# -*- coding: utf-8 -*-py
import os 
import nltk
from os import listdir
from lexical_simplification.recursive_simplification import RecursiveSimplification

# To be run from ./exp_threshold/ directory
RECURSIVE_CONFIG_PATH = './recursive_config_threshold.yaml'
TXT_PATHS = listdir('../onestop/adv-text/')
TXT_PATHS = ['../onestop/adv-text/' + filename for filename in TXT_PATHS]

if __name__ == '__main__':
    recursive_simplification = RecursiveSimplification(recursive_config_path=RECURSIVE_CONFIG_PATH)
    thresholds = [0.5, 0.8, 1.0]

    for t in thresholds:
        recursive_simplification.set_param(param='threshold', value=t)
        for txt_path in TXT_PATHS:
            base = os.path.basename(txt_path)
            filename_without_ext = os.path.splitext(base)[0]
            simplified_save_path = '{0}/{1}-{2}.0.txt'.format('../onestop/simplified', filename_without_ext,
                                                              int(10*t))

            f = open(simplified_save_path, "w+")
            lines = open(txt_path, 'r').readlines()
            for line in lines:
                tokenized_text = nltk.word_tokenize(line)
                simplified_tokens = recursive_simplification.apply_simplification(tokenized_text=tokenized_text)
                f.write('{0}\n'.format(' '.join(simplified_tokens)))
            f.close()

