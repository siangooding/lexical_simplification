# -*- coding: utf-8 -*-py
import argparse
from readability_code.main_extract_features import analyze_files

READABILITY_CODE_PATH = '/home/ib431/Documents/projects/cam_mphil_project/readability_code/'
NEW_LEVELS = {'1': 1, '2': 2, '3': 3}


if __name__ == '__main__':
    # To be run from ./final_exp_simplify/ folder
    # python run_pre_cache_features.py -i test_folder/train/ -o train_feats_test -f T_L_SGR_NER
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input_folder", required=True, help="input folder to take the files from" +
                                                                "must be a train folder")
    ap.add_argument("-o", "--output_file", required=True, help="output name for the stored features" +
                                                                "if file already exists will be overwritten")
    ap.add_argument("-f", "--features", required=True, help="features to consider, separated by `_`" +
                                                            "currently implemented: T_L_SGR_NER")
    args = vars(ap.parse_args())

    analyze_files(main_path=READABILITY_CODE_PATH,
                  inputdir='/home/ib431/Documents/projects/cam_mphil_project/readability_code/OneStop-modified/'+args["input_folder"],
                  outdir="/home/ib431/Documents/projects/cam_mphil_project/readability_code/OneStop_features", levels=NEW_LEVELS,
                  features=args["features"], output_file=args["output_file"], is_test=False)