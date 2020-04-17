# -*- coding: utf-8 -*-py
import nltk
import csv
import pickle
from os import listdir
from collections import defaultdict
from nltk.corpus import stopwords
import xml.etree.ElementTree as ET

stop_words = set(stopwords.words('english'))
punctuation = set(['.', ',', ';', ':', '"', "'", '`', '?', '!'])
TO_DISCARD  = stop_words.union(punctuation)


def get_key_val(row, tsv_path):
    if "CEFR_levels" in tsv_path:
        key, val = row['a'], row['1']
    elif "KET_list" in tsv_path:
        key, val = row["'ll"], 2
    else: 
        key, val = row["1"], 3
    return key, val


def is_string_of_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def get_text_from_xml(xml_path):
    for _, element in ET.iterparse(xml_path):
        if element.tag == "text":
            return element.text
    # No text content (security case)
    return None


def update_word_count(text, parser, word_count=defaultdict(int), tot_count=0):
    words = nltk.word_tokenize(text)
    words = [word for word in words if word not in TO_DISCARD]
    for word in words:
        parser_output = parser.nlp([[word]])
        word_count[parser_output.sentences[0].words[0].lemma] += 1
        tot_count += 1
    return word_count, tot_count


def save_frequency_info(parser, levelled_text_folder='./readability_code/CEPP-rhaps/',
                        save_path='./cefr_freq_info.pkl'):
    stored_info = {"word_count": {}, "tot_count": {}}
    levels = listdir(levelled_text_folder)
    levels = [level for level in levels if len(level) == 3]
    for level in levels:
        curr_word_count, curr_tot_count = defaultdict(int), 0
        files = listdir(levelled_text_folder+level)
        files = [levelled_text_folder+level+'/'+name for name in files]

        for file_path in files:
            text = get_text_from_xml(xml_path=file_path)
            curr_word_count, curr_tot_count = update_word_count(text=text,
                                                                parser=parser,
                                                                word_count=curr_word_count,
                                                                tot_count=curr_tot_count)
        stored_info["word_count"][level] = curr_word_count
        stored_info["tot_count"][level] = curr_tot_count

    with open(save_path, 'wb') as f:
        pickle.dump(stored_info, f)



if __name__ == '__main__':
    from stanfordnlp_.stanfordnlp_parser import StanfordNLP
    # save_frequency_info(parser=StanfordNLP())
    frequency_info = pickle.load(open('./cefr_freq_info.pkl', 'rb'))
    print(frequency_info["tot_count"])
    print(frequency_info["word_count"]["CAE"].keys())
    for level in frequency_info["word_count"]:
        one_occ = {key:value for key, value in frequency_info["word_count"][level].items() if value == 6}
        counts = set([value for _, value in frequency_info["word_count"][level].items()])
        print(level, counts)
        print(level, len(frequency_info["word_count"][level]), len(one_occ))