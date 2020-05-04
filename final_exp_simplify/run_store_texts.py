# -*- coding: utf-8 -*-py
import os
import nltk
import yaml
import pickle
import argparse
import subprocess
from os import listdir
from datetime import datetime
from lexical_simplification.recursive_simplification import RecursiveSimplification

# One unique config files for all experiments run : to be changed each time
RECURSIVE_CONFIG_PATH = '/home/ib431/Documents/projects/cam_mphil_project/final_exp_simplify/recursive_config_exp.yaml'
with open(RECURSIVE_CONFIG_PATH) as file:
    CONFIG = yaml.load(file, Loader=yaml.FullLoader)

INPUT_ROOT = '/home/ib431/Documents/projects/cam_mphil_project/onestop/adv-text-all/'
INPUT_FILES = listdir(INPUT_ROOT)
OUTPUT_ROOT = '/home/ib431/Documents/projects/cam_mphil_project/data/{0}/'
LOG_PICKLE = '/home/ib431/Documents/projects/cam_mphil_project/data/{0}/log.pkl'
F_SAVE = '/home/ib431/Documents/projects/cam_mphil_project/data/{0}/process_pipeline.txt'

TYPES = {'baseline': 'baseline_{0}',
         'no_salience': '{0}_no_salience_{1}_{2}_{3}',
         'cascade': '{0}_salience_cascade_{1}_{2}_{3}_{4}',
         'combined': '{0}_salience_combined_{1}_{2}_{3}_{4}_{5}'}

def get_folder_name(type_system, config=CONFIG):
    if type_system == 'baseline':
        return TYPES[type_system].format(CONFIG['threshold'])
    elif type_system == 'no_salience':
        return TYPES[type_system].format(CONFIG['level_target'], CONFIG['threshold'],
                                         CONFIG['gamma'], CONFIG['delta'])
    elif type_system == 'cascade':
        return TYPES[type_system].format(CONFIG['level_target'],
                                         CONFIG['salience_threshold'], CONFIG['threshold'],
                                         CONFIG['gamma'], CONFIG['delta']) 
    else:  # type_system = combined
        return TYPES[type_system].format(CONFIG['level_target'],
                                         CONFIG['alpha'], CONFIG['beta'],
                                         CONFIG['global_threshold'],
                                         CONFIG['gamma'], CONFIG['delta']) 


if __name__ == '__main__':
    # To be run from ./final_exp_simplify/ folder
    # python run_store_texts.py -t baseline
    # python run_store_texts.py -t no_salience
    # python run_store_texts.py -t cascade
    # python run_store_texts.py -t combined
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--type_system", required=True, help="Type of system to use to simplify the texts" + 
                                                               "Must be one key of the `TYPES` dictionary")

    # @type_system==`baseline`: using the original recursive simplification system developed by Sian. Only difference is
    # some new resources for synonym generation were added. `include_salience` is False and only the `threshold` needs to be set.
    # @type_system==`no_salience`: just using the pre selection score. `include_salience` is False.
    # `threshold`, `gamma` and `delta` must be non null.
    # @type_system==`cascade`: using pre selection score + combined score with cascade type. `include_salience` is True,
    # `type_imp` is `cascade`. `salience_threshold`, `threshold`, `gamma` and `delta` must be non null.
    # @type_system==`combined`: using pre selection score + combined score with combined type. `include_salience` is True,
    # `type_imp` is `combined`. `alpha`, `beta`, `global_threshold`, `gamma` and `delta` must be non null.

    args = vars(ap.parse_args())
    recursive_simplification = RecursiveSimplification(recursive_config_path=RECURSIVE_CONFIG_PATH)
    folder_name = get_folder_name(type_system=args["type_system"])
    output_root = OUTPUT_ROOT.format(folder_name)
    if not os.path.exists(output_root):
        subprocess.call("mkdir {0}".format(output_root), shell=True)
    recursive_simplification.set_param(param='f_save', value=F_SAVE.format(folder_name))

    logs = {}

    for filename in INPUT_FILES:
        if not os.path.exists(output_root + filename):
            date_begin = datetime.now()
            curr_tokens = nltk.word_tokenize(''.join(open(INPUT_ROOT+filename, 'r').readlines()))
            curr_simplified = recursive_simplification.apply_simplification(tokenized_text=curr_tokens)

            f = open(output_root + filename, "w+")
            f.write(' '.join(curr_simplified))
            f.close()
            date_end = datetime.now()
            logs[filename] = {'begin': date_begin, 'end': date_end}
    
    pickle.dump(logs, open(LOG_PICKLE.format(folder_name), 'wb'))
