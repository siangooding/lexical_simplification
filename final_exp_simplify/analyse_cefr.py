# -*- coding: utf-8 -*-py
import pickle
import argparse
import numpy as np
from os import listdir
from settings.settings import ROOT_PATH

REL_PATH = 'final_exp_simplify/cefr_ls_res/'


def get_prop_score(data):
    key_print = [('nb_gen_syn_pos', 'positive'), ('nb_gen_syn_zero', 'zero'), ('nb_gen_syn_neg', 'negative')]
    nb_syn = [elt for elt in data['nb_gen_syn'] if elt is not None]
    for (key, print_val) in key_print:
        curr_data = [elt for elt in data[key] if elt is not None]
        curr_data = [round(100 * curr_data[i] / nb_syn[i], 2) for i in range(len(curr_data))]
        print('Average % of synonyms with {0} score: {1}'.format(print_val, round(np.mean(np.array(curr_data)), 2)))


def get_neg_info(data):
    key_print=  [('neg_target_level', 'at least one synonym with a negative score in the annotated target level'),
                 ('neg_target_level_lower', 'at least one synonym with a negative score in the annotated target and lower levels')]
    for (key, print_val) in key_print:
        curr_data = [elt for elt in data[key] if elt is not None]
        print('% of words with {0}: {1}'.format(print_val, round(100 * len([elt for elt in curr_data if elt > 0])/len(curr_data), 2)))


def get_top_info(data):
    key_print=  [('top_5_target_level', 'at least one word in the top 5 in the annotated target level'),
                 ('top_5_target_level_lower', 'at least one word in the top 5 in the annotated target and lower levels'),
                 ('top_5_higher_level', 'at least one word in the top 5 in the annotated higher levels')]
    for (key, print_val) in key_print:
        curr_data = [elt for elt in data[key] if elt is not None]
        print('% of words with {0}: {1}'.format(print_val, round(100 * len([elt for elt in curr_data if elt > 0])/len(curr_data), 2)))


if __name__ == '__main__':
    """ Analysing results using different parameters for the pre selection score"""
    # python analyse_cefr.py -l 2
    ap = argparse.ArgumentParser()
    ap.add_argument("-l", "--level", required=True, help="which level (either 2 or 3) to analyse")
    args = vars(ap.parse_args())

    files = [file_name for file_name in listdir(ROOT_PATH+REL_PATH) if file_name[0] == args["level"]]
    for i, file_name in enumerate(files):
        data = pickle.load(open(ROOT_PATH+REL_PATH+file_name, 'rb'))
        if i == 0:
            print('General results for level 2')
            print('Average # of A1 simplifications: {0}'.format(round(np.mean(np.array([elt for elt in data['len_a1'] if elt is not None])), 1)))
            print('Average # of A2 simplifications: {0}'.format(round(np.mean(np.array([elt for elt in data['len_a2'] if elt is not None])), 1)))
            print('Average # of B1 simplifications: {0}'.format(round(np.mean(np.array([elt for elt in data['len_b1'] if elt is not None])), 1)))
            print('Average # of synonyms per word: {0}'.format(round(np.mean(np.array([elt for elt in data['nb_gen_syn'] if elt is not None])), 1)))
            get_prop_score(data=data)
            get_neg_info(data=data)
            print('====================')

        print('Analyzing {0}'.format(file_name))
        get_top_info(data=data)
        print('====================')
        
