# -*- coding: utf-8 -*-
import argparse
import ntpath

def sep_levels(aligned_path, save_folder):
    """ From aligned sentences in one file
    Create two distinct files 
    If aligned file name like A-B, then first path corresponds
    to A level, and second path to B level """
    [aligned_name, _] = ntpath.basename(aligned_path).split('.')
    [first, second] = aligned_name.split('-')

    first_path = '{0}onestop.en-en.{1}'.format(save_folder, first.lower())
    second_path = '{0}onestop.en-en.{1}'.format(save_folder, second.lower())

    aligned_lines = open(aligned_path, encoding="utf-8").readlines()
    first_file = open(first_path,"w+", encoding="utf-8")
    second_file = open(second_path, 'w+', encoding="utf-8")

    for index, line in enumerate(aligned_lines):
        if index % 3 == 0:  # first level 
            first_file.write(line)
        if index % 3 == 1:  # second level
            second_file.write(line)
    
    first_file.close()
    second_file.close()


def reverse_file(input_path, save_path):
    lines = open(input_path).readlines()
    f_save = open(save_path, "w+")
    
    for i in range(len(lines)//3):
        ele_, int_, star_ = i * 3, i * 3 + 1, i * 3 + 2
        f_save.write(lines[int_])
        f_save.write(lines[ele_])
        f_save.write(lines[star_])
    f_save.close()


if __name__ == '__main__':
    # Should be run from exp_alignment/ folder from root directory, otherwise change relative paths
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="Input file - Aligned sentences ")
    ap.add_argument("-o", "--output", required=True, help="Output folder - Save folder for both files ")
    args = vars(ap.parse_args())

    sep_levels(aligned_path=args["input"], 
               save_folder=args["output"])
    
    # For the OneStopEnglish aligned corpus
    # python create_sep_file.py -i ../onestop/aligned/ADV-ELE.txt -o ./giza-pre-process/
    # python create_sep_file.py -i ../onestop/aligned/ADV-INT.txt -o ./giza-pre-process/
    # python create_sep_file.py -i ../onestop/aligned/ELE-INT.txt -o ./giza-pre-process/

    # input_path = '../onestop/aligned/ELE-INT.txt'
    # save_path = '../onestop/aligned/INT-ELE.txt'
    # reverse_file(input_path, save_path)