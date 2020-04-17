# -*- coding: utf-8 -*-
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))
from find_candidates import find_info, get_lemmas_per_sent, validate, validate_single


def get_lines(file_path):
    return open(file_path).readlines()


def compare_lines(file1, file2):
    lines_1 = get_lines(file1)
    lines_2 = get_lines(file2)
    common = 0
    for i, line_1 in enumerate(lines_1):
        if lines_2[i] == line_1:
            common += 1
    print("Common lines between {0} and {1} \n {2}".format(file1, file2, common))


def validate_null(indexes, info):
    for index in indexes:
        if info[index] not in STOPWORDS:
            return True
    return False


def get_numbers(giza_output_path, source_nlp_path, target_nlp_path, type_):
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

    rewordings, deletions, splits, merges = 0, 0, 0, 0
    for sent_pair_index, tokens_target in enumerate(target):
        # Iterating through all paired sentences in GIZA++ output
        source_words = source[sent_pair_index]
        # First analysis with the `NULL` part, i.e. deletions
        if type_ == "reversed":
            found_deletions = True
            if validate_null(indexes=source_words[0][1], info=tokens_target):
                deletions += 1
        else:
            found_deletions = False
        
        found_rewording, found_split, found_merge = False, False, False
        merge_track = [0] * len(tokens_target)
        for source_index, (_, indexes) in enumerate(source_words):
            if (source_index != 0) and (len(indexes) == 1):  # one-to-one correspondence
                curr_lemma_source = source_lemmas[sent_pair_index][source_index].attrib
                curr_lemma_target = target_lemmas[sent_pair_index][indexes[0]].attrib

                if not found_rewording:
                    if validate(curr_lemma_source, curr_lemma_target):  # pair responds to criteria
                        rewordings += 1
                        found_rewording = True
                
                if not found_merge:
                    if validate_single(curr_lemma_target):
                        if merge_track[indexes[0]] == 1:
                            merges += 1
                            found_merge = True
                        else:
                            merge_track[indexes[0]] += 1
            
            if (source_index != 0) and (len(indexes) > 1) and (not found_split):
                curr_lemma_source = source_lemmas[sent_pair_index][source_index].attrib
                if validate_single(curr_lemma_source):
                    splits += 1
                    found_split = True
            
            if (source_index != 0) and (len(indexes) == 0) and (not found_deletions):
                curr_lemma_source = source_lemmas[sent_pair_index][source_index].attrib
                if validate_single(curr_lemma_source):
                    deletions += 1
                    found_deletions = True

    print("Number of re-wordings found: {0}".format(rewordings))
    print("Number of deletions found: {0}".format(deletions))
    print("Number of splits found: {0}".format(splits))
    print("Number of merges found: {0}".format(merges))



if __name__ == '__main__':
    COMPARE_LINES = False
    if COMPARE_LINES:
        file1 = './giza-pre-process/adv-ele/onestop.en-en.adv'
        file2 = './giza-pre-process/adv-ele/onestop.en-en.ele'
        compare_lines(file1, file2)

        file1 = './giza-pre-process/adv-int/onestop.en-en.adv'
        file2 = './giza-pre-process/adv-int/onestop.en-en.int'
        compare_lines(file1, file2)

        file1 = './giza-pre-process/ele-int/onestop.en-en.ele'
        file2 = './giza-pre-process/ele-int/onestop.en-en.int'
        compare_lines(file1, file2)

    print("Numbers for ADV-INT alignments")
    GIZA_OUTPUT_PATH = './giza-output/adv-int/2020-03-25.100118.ib431.VA3.final'
    SOURCE_NLP_PATH = './giza-pre-process/adv-int/onestop.en-en.tok.adv.xml.stanfordnlp'
    TARGET_NLP_PATH = './giza-pre-process/adv-int/onestop.en-en.tok.int.xml.stanfordnlp'
    get_numbers(giza_output_path=GIZA_OUTPUT_PATH,
                source_nlp_path=SOURCE_NLP_PATH,
                target_nlp_path=TARGET_NLP_PATH,
                type_="normal")
    print("==========")

    print("Numbers for ADV-ELE alignments")
    GIZA_OUTPUT_PATH = './giza-output/adv-ele/2020-03-25.095820.ib431.VA3.final'
    SOURCE_NLP_PATH = './giza-pre-process/adv-ele/onestop.en-en.tok.adv.xml.stanfordnlp'
    TARGET_NLP_PATH = './giza-pre-process/adv-ele/onestop.en-en.tok.ele.xml.stanfordnlp'
    get_numbers(giza_output_path=GIZA_OUTPUT_PATH,
                source_nlp_path=SOURCE_NLP_PATH,
                target_nlp_path=TARGET_NLP_PATH,
                type_="normal")
    print("=====")
    

    print("Numbers for INT-ELE alignments")
    GIZA_OUTPUT_PATH = './giza-output/int-ele/2020-04-03.133908.ib431.VA3.final'
    SOURCE_NLP_PATH = './giza-pre-process/int-ele/onestop.en-en.tok.int.xml.stanfordnlp'
    TARGET_NLP_PATH = './giza-pre-process/int-ele/onestop.en-en.tok.ele.xml.stanfordnlp'
    get_numbers(giza_output_path=GIZA_OUTPUT_PATH,
                source_nlp_path=SOURCE_NLP_PATH,
                target_nlp_path=TARGET_NLP_PATH,
                type_="normal")
    print("==========")