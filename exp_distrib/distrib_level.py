# -*- coding: utf-8 -*-
import os
import re
import pickle
import argparse
import xml.etree.ElementTree as ET
import nltk
from collections import defaultdict
from lexical_simplification.simplification import Sentence
from lexical_simplification.scores import ImportanceSimplifyScore

NLP = None
IMPORTANCE_SCORE = ImportanceSimplifyScore(include_salience=False, threshold=0.5)


def pre_process(text):
    text = re.sub("\n"," ", text)
    text = re.sub("\"","\"", text)
    text = re.sub("''","\"", text)
    text = re.sub("``","\"", text)	
    text = re.sub(" +"," ", text)
    return text


def get_complexities(distrib, level, file_path):
    text = None
    for _, element in ET.iterparse(file_path):
        if element.tag == 'text':
            text = pre_process(text=element.text)
    
    if text is not None:
        # Text-level distribution
        text_tokenized = nltk.word_tokenize(text)
        text_object = Sentence(tokenized=text_tokenized,
                               threshold=0,
                               ignore_list=[],
                               nlp=NLP, importance_score=IMPORTANCE_SCORE)
        for (index, complexity) in text_object.complex_words:
            distrib[level]['text'][text_tokenized[index]].append(complexity)
        
        # Sentence-level distribution
        sentence_token = nltk.data.load('tokenizers/punkt/english.pickle')
        lines = sentence_token.tokenize(text.strip())
        for line in lines:
            sent_tokenized = nltk.word_tokenize(line)
            sent_object = Sentence(tokenized=sent_tokenized,
                                   threshold=0,
                                   ignore_list=[],
                                   nlp=NLP, importance_score=IMPORTANCE_SCORE)
            for (index, complexity) in sent_object.complex_words:
                distrib[level]['sent'][sent_tokenized[index]].append(complexity)
    
    return distrib


def main(all_text_level_path, save_path):
    distrib = {}
    for level in os.listdir(all_text_level_path):
        if level not in ['.DS_Store']:
            distrib[level] = {
                'sent': defaultdict(list),
                'text': defaultdict(list)
            }
            for file_path in os.listdir(all_text_level_path+level):
                distrib = get_complexities(distrib, level,
                                           file_path='{0}{1}/{2}'.format(
                                               all_text_level_path,
                                               level,
                                               file_path))
    
    with open(save_path, 'wb') as outfile:
        pickle.dump(distrib, outfile)


if __name__ == '__main__':
    # Assumed run from exp_distrib folder root directory
    # python distrib_level.py -i ../readability_code/CEPP-rhaps/ -o ./distrib.pickle
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="all texts level path ")
    ap.add_argument("-o", "--output", required=True, help="saving path, .pickle file ")
    args = vars(ap.parse_args())

    main(all_text_level_path=args['input'],
         save_path=args['output'])