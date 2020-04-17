# -*- coding: utf-8 -*-
import re
import pickle
import argparse
from collections import defaultdict
from nltk.corpus import stopwords
import xml.etree.ElementTree as ET

STOP_WORDS = list(stopwords.words('english'))
NO_WORDS = ['NULL']
PUNCTUATION = [',','?', ';', '.', ':', '/', '!']
ALL_STOP_WORDS = STOP_WORDS + NO_WORDS + PUNCTUATION

POS_PROPER_NOUN = ['NP', 'NP1', 'NP2', 'NPD1', 'NPD2', 'NPM1', 'NPM2']
UPOS_NO_LIST = ['PROPN', 'NUM', 'PRON']



def find_info(concat):
    word = concat.split(' ')[0]
    try:
        indexes = re.findall("({\\s(.*?)\\s)}", concat)[0][1].split(' ')
    except Exception:
        indexes = []
    return word, [int(index) for index in indexes]


def get_lemmas_per_sent(nlp_path):
    root = ET.parse(nlp_path).getroot()
    lemma_list = root.findall('./stanfordnlp/sentence/word-list')
    lemmas_per_sent = [[''] + lemmas.findall('word') for lemmas in lemma_list]
    return lemmas_per_sent


def validate_single(info):
    if info["text"] in ALL_STOP_WORDS:
        return False
    if info["upos"] in UPOS_NO_LIST:
        return False
    return True


def validate_double(source, target):
    if source["lemma"] == target["lemma"]:
        return False
    if source["upos"] != target["upos"]:
        return False
    return True


def validate(lemma_source, lemma_target):
    """ Validate pairing of two words according to following criteria :
    Not in stoplist, same POS, no proper noun, different lemmas """
    # Example of possible lemmas (from Stanford NLP output, converted to XML)
    # Cf. ../stanfordnlp_/stanfordnlp_parser.py for more info
    # <word index="20" lemma="by" text="by" upos="ADP" xpos="IN" />
    return validate_single(lemma_source) and validate_single(lemma_target) and \
    validate_double(lemma_source, lemma_target)


def extract_candidates(giza_output_path, source_nlp_path, target_nlp_path, save_path):
    """ From the output of GIZA++ extract candidates
    for simpler alternatives 
    Format chosen : lines beginning with NULL : higher level sentences 
    Source = higher level (sent with annotations)
    Target = lower level (raw sentence) """
    lines_giza = open(giza_output_path).readlines()
    lines_giza = [line for index, line in enumerate(lines_giza) if index % 3 != 0]
    lines_giza = [line.replace('\n', '') for line in lines_giza]

    source = [line.split('}) ') for index, line in enumerate(lines_giza) if index % 2 == 1]
    source = [[''.join([elt, '})']) for elt in line ] for line in source ]
    source = [[find_info(elt) for elt in line][:-1] for line in source]
    # source[0], i.e. first source sentence on file, like 
    # [('NULL', []), ('The', []), ('Seattle-based', [1]), ('company', []), ('has', [2])]
    target = [[''] + line.split(' ') for index, line in enumerate(lines_giza) if index % 2 == 0]
    # target[0], i.e. first target sentence on file, like 
    # ['', 'Amazon', 'has', 'asked', 'for', 'its', 'company']

    source_lemmas = get_lemmas_per_sent(nlp_path=source_nlp_path)
    target_lemmas = get_lemmas_per_sent(nlp_path=target_nlp_path)

    candidates = defaultdict(list)
    for sent_pair_index, _ in enumerate(target):
        # Iterating through all paired sentences in GIZA++ output
        source_words = source[sent_pair_index]
        for source_index, (_, indexes) in enumerate(source_words):
            if (source_index != 0) and (len(indexes) == 1):  # one-to-one correspondence
                curr_lemma_source = source_lemmas[sent_pair_index][source_index].attrib
                curr_lemma_target = target_lemmas[sent_pair_index][indexes[0]].attrib
                if validate(curr_lemma_source, curr_lemma_target):  # pair responds to criteria
                    source_lem, target_lem = curr_lemma_source["lemma"], curr_lemma_target["lemma"]
                    upos = curr_lemma_source["upos"]
                    if target_lem not in candidates['{0}_{1}'.format(source_lem, upos)]:
                        candidates['{0}_{1}'.format(source_lem, upos)].append(target_lem)
    
    candidates = dict((key, value) for key, value in candidates.items() if value != [])

    with open(save_path, 'wb') as f:
        pickle.dump(candidates, f)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-g", "--giza", required=True, help="Giza output .VA3.final file name format ")
    ap.add_argument("-so", "--source", required=True, help="Tokenized sentences for source sentences ")
    ap.add_argument("-t", "--target", required=True, help="Tokenized sentences for target sentences ")
    ap.add_argument("-sa", "--save", required=True, help="Save path for candidates - .pkl file ")
    args = vars(ap.parse_args())

    extract_candidates(giza_output_path=args["giza"], source_nlp_path=args["source"],
                       target_nlp_path=args["target"], save_path=args["save"])

    # For aligned corpus and GIZA++
    # python 
    # find_candidates.py 
    # -g ./giza-output/2020-03-20.110444.ib431.VA3.final 
    # -so ./giza-pre-process/onestop.en-en.tok.adv.xml.stanfordnlp
    # -t ./giza-pre-process/onestop.en-en.tok.ele.xml.stanfordnlp 
    # -sa ./levelled_words/adv-ele.pkl

    # Reversing ele-int to int-ele to have for all files
    # Key = higher level, value = lower level
    # import pickle
    # from collections import defaultdict
    # path = './levelled_words/ele-int.pkl'
    # data = pickle.load(open(path, 'rb'))

    # new_data = defaultdict(list)
    # for key, value in data.items():
    #     for upper_word in  value:
    #         sep = key.split('_')
    #         complex_word = ''.join(sep[:-1])
    #         upos = sep[-1]
    #         new_data['{0}_{1}'.format(upper_word, upos)].append(complex_word)
    # pickle.dump(new_data, open('./levelled_words/int-ele.pkl', 'wb'))