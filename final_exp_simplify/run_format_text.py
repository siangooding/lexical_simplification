# -*- coding: utf-8 -*-py
import os
import yaml
import subprocess
from os import listdir
from format_input.format_input_xml import create_complete_xml

OUTPUT_ROOT = '/home/ib431/Documents/projects/cam_mphil_project/readability_code/OneStop-modified/'

FORMAT_CONFIG_PATH = '/home/ib431/Documents/projects/cam_mphil_project/final_exp_simplify/format_config.yaml'
with open(FORMAT_CONFIG_PATH) as file:
    CONFIG = yaml.load(file, Loader=yaml.FullLoader)
INDEXES_TRAIN = CONFIG['indexes']['train']
INDEXES_TEST = CONFIG['indexes']['test']

def create_dir(path):
    if not os.path.exists(path):
        subprocess.call("mkdir {0}".format(path), shell=True)
        for type_folder in ['train', 'test']:
            subprocess.call("mkdir {0}/{1}".format(path, type_folder), shell=True)
            for level in [1, 2, 3]:
                subprocess.call("mkdir {0}/{1}/{2}".format(path, type_folder, level), shell=True)

def get_files_name(path):
    files = listdir(path)
    return [file for file in files if file not in ['log.pkl', 'process_pipeline.txt']]


if __name__ == '__main__':
    # To be run from ./final_exp_simplify/ folder
    # python run_format_text.py
    # Creating directories to store data 
    create_dir(path=OUTPUT_ROOT+CONFIG['folder'])

    KEY_TO_LEVEL = {'lower_level_folder': 1, 'medium_level_folder': 2, 'higher_level_folder': 3}
    for key, level in KEY_TO_LEVEL.items():
        filenames = get_files_name(path=CONFIG[key])
        for index, filename in enumerate(filenames):
            if index in INDEXES_TRAIN:
                create_complete_xml(text_path=CONFIG['lower_level_folder']+filename,
                                    folder_xml="{0}/{1}/{2}/".format(OUTPUT_ROOT+CONFIG['folder'], 'train', level))
            if index in INDEXES_TEST:
                create_complete_xml(text_path=CONFIG['lower_level_folder']+filename,
                                    folder_xml="{0}/{1}/{2}/".format(OUTPUT_ROOT+CONFIG['folder'], 'test', level))



