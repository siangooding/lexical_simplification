# -*- coding: utf-8 -*-
from os import listdir
import argparse
import subprocess

"""
To run for three types of aligned sentences
ADV (adv) - INT(int) - ELE(ele)
1) ADV-ELE
2) ADV-INT
3) ELE-INT
"""

def find_va3_file(files_list):
    for file in files_list:
        if ".VA3.final" in file:
            return file
    return ''


def run_script(first, second):
    """
    (first, second) must be one of the followings:
    (adv, ele), (adv, int), (ele, int)
    """
    sep_file = "python create_sep_file.py -i ../onestop/aligned/{0}-{1}.txt -o ./giza-pre-process/{2}-{3}/"
    subprocess.call(sep_file.format(first.upper(), second.upper(), first, second), shell=True)
    print("Created two separate files with distinct levels")

    pre_giza = "./mosesdecoder/scripts/tokenizer/tokenizer.perl -l en < ./giza-pre-process/{0}-{1}/onestop.en-en.{2} > ./giza-pre-process/{0}-{1}/onestop.en-en.tok.{2}"
    subprocess.call(pre_giza.format(first, second, first), shell=True)
    subprocess.call(pre_giza.format(first, second, second), shell=True)
    print("Tokenized sentences from both levels")

    snt_vsb = "./giza-pp-master/GIZA++-v2/plain2snt.out ./giza-pre-process/{0}-{1}/onestop.en-en.tok.{0} ./giza-pre-process/{0}-{1}/onestop.en-en.tok.{1}"
    cooc = "./giza-pp-master/GIZA++-v2/snt2cooc.out ./giza-pre-process/{0}-{1}/onestop.en-en.tok.{0}.vcb ./giza-pre-process/{0}-{1}/onestop.en-en.tok.{1}.vcb  ./giza-pre-process/{0}-{1}/onestop.en-en.tok.{0}_onestop.en-en.tok.{1}.snt > ./giza-pre-process/{0}-{1}/corp-{0}-{1}.cooc"
    subprocess.call(snt_vsb.format(first, second), shell=True)
    subprocess.call(cooc.format(first, second), shell=True)
    print("Created three additional files necessary to run GIZA++")

    giza = "./giza-pp-master/GIZA++-v2/GIZA++ -S ./giza-pre-process/{0}-{1}/onestop.en-en.tok.{0}.vcb -T ./giza-pre-process/{0}-{1}/onestop.en-en.tok.{1}.vcb -C ./giza-pre-process/{0}-{1}/onestop.en-en.tok.{0}_onestop.en-en.tok.{1}.snt -CoocurrenceFile ./giza-pre-process/{0}-{1}/corp-{0}-{1}.cooc -outputpath ./giza-output/{0}-{1}/"
    subprocess.call(giza.format(first, second), shell=True)
    print("Ran GIZA++")

    pos = "python ../stanfordnlp_/stanfordnlp_parser.py -i ./giza-pre-process/{0}-{1}/onestop.en-en.tok.{2} -o ./giza-pre-process/{0}-{1}/"
    subprocess.call(pos.format(first, second, first), shell=True)
    subprocess.call(pos.format(first, second, second), shell=True)
    print("Applied StanfordNLP Parser and create XML files")

    files_list = listdir("./giza-output/{0}-{1}".format(first, second))
    va3_file = find_va3_file(files_list)
    candidates = "python find_candidates.py -g ./giza-output/{0}-{1}/{2} -so ./giza-pre-process/{0}-{1}/onestop.en-en.tok.{0}.xml.stanfordnlp -t ./giza-pre-process/{0}-{1}/onestop.en-en.tok.{1}.xml.stanfordnlp -sa ../onestop/aligned/{0}-{1}.pkl"
    subprocess.call(candidates.format(first, second, va3_file), shell=True)
    print("Found and stored candidates")


if __name__ == '__main__':
    """ To be run from `exp_alignment` directory """
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--first", required=True, help="First level to consider")
    ap.add_argument("-s", "--second", required=True, help="Second level to consider")
    args = vars(ap.parse_args())

    run_script(first=args["first"], second=args["second"])

    # For aligned corpus and GIZA++
    # python run_giza.py -f adv -s ele
    # python run_giza.py -f adv -s int
    # python run_giza.py -f ele -s int

    # For common advanced sentences
    # Removing creating two separate files part 
    # python run_giza.py -f adv -s ele-common
    # python run_giza.py -f adv -s int-common