# -*- coding: utf-8 -*-py
import csv
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import multiprocessing
import gensim.downloader as api
from collections import defaultdict
from settings.settings import ROOT_PATH
from allennlp.commands.elmo import ElmoEmbedder
from stanfordnlp_.stanfordnlp_parser import StanfordNLP
from lexical_simplification.simplification import Sentence, Word
from lexical_simplification.scores import ImportanceSimplifyScore, SelectionScore

NLP = StanfordNLP()
NLP = NLP.nlp
CACHED_SYN_PATH = ROOT_PATH + 'lexical_simplification/cached_synonyms.yaml'
ZERO_SCORE_PATH = ROOT_PATH + 'lexical_simplification/zero_score.txt'
ELMO = ElmoEmbedder()
GLOVE_MODEL = api.load('glove-wiki-gigaword-50')
WORD2VEC_MODEL = api.load('word2vec-google-news-300')

# MAPPING_SENT_INDEX_TO_WORD_INDEX = {
#     62: 12, 69: 27, 71: 11, 95: 13, 112: 20, 175: 6, 180: 2, 204: 19, 207: 0, 208: -1,
#     270: -1, 284: -1, 300: -1, 311: -1, 318: -1, 324: -1, 345: -1, 353: -1, 358: -1, 367: -1, 368: -1,
#     381: -1, 387: -1, 402: -1, 414
# }

class SentenceCEFRLS:
    def __init__(self, list_tsv, include_salience, threshold,
                 alpha, beta, salience_threshold, global_threshold, type_imp,
                 level_target, gamma, delta):
        self.list_tsv = list_tsv
        row = [elt for elt in self.list_tsv if elt not in ['0', '1']]

        self.tokenized_text = row[0].split(' ')
        self.tokenized_text = [elt for elt in self.tokenized_text if elt != '']
        self.complex_word = row[1]
        self.index = self.find_index(self.tokenized_text, self.complex_word)
        self.levelled_syn, self.level_word = self.get_info_from_tsv_line(row)
        
        self.importance_score = ImportanceSimplifyScore(include_salience, alpha, beta,
                                                        threshold, salience_threshold,
                                                        global_threshold, type_imp)
        self.level_target = level_target
        self.selection_score = SelectionScore(level_target, gamma, delta, NLP)
        self.sentence_object = Sentence(self.tokenized_text, 0, [], NLP, self.importance_score)
        if self.index is not None:
            self.word_object = Word(self.sentence_object, self.index, CACHED_SYN_PATH, ZERO_SCORE_PATH,
                                    NLP, ELMO,
                                    GLOVE_MODEL, WORD2VEC_MODEL)
        else:
            self.word_object = None
    
    def find_index(self, tokenized_text, complex_word):
        for index, elt in enumerate(tokenized_text):
            if elt.lower() == complex_word.lower():
                return index
        # lemma different from morphological variant in the text
        for index, elt in enumerate(tokenized_text):
            if NLP([[elt]]).sentences[0].words[0].lemma == complex_word:
                return index
        return None
    
    def get_info_from_tsv_line(self, row):
        levelled_syn = defaultdict(list)
        level_word = row[2]
        for i in range(len(row[3:])//2):
            levelled_syn[row[3:][2*i + 1]].append(row[3:][2*i])
        return levelled_syn, level_word
    
    def get_score(self):
        for (index, complexity) in self.sentence_object.complex_words:
            if index == self.index:
                return complexity
        return None
    
    def get_synonyms(self):
        self.word_object.get_synonyms()
        if self.level_target:  # selection
            synonyms = [syn[0] for syn in self.word_object.synonyms]
            scored = [(word, self.selection_score.get_pre_selection_score(word)) for word in synonyms]
            scored = sorted(scored, key=lambda tup: tup[1], reverse=True)
            return scored


class SentencesCEFRLS:
    def __init__(self, tsv_path=ROOT_PATH + 'final_exp_simplify/CEFR-LS.tsv', include_salience=False, threshold=0,
                 alpha=None, beta=None, salience_threshold=None, global_threshold=0, type_imp=None,
                 level_target=None, gamma=None, delta=None):
        read_tsv = csv.reader(open(tsv_path), delimiter="\t")
        self.nb_to_cefr = {1: 'A1', 2: 'A2', 3: 'B1', 4: 'B2', 5: 'C1', 6: 'C2'}
        self.cefr_to_nb = {'A1': 1, 'A2': 2, 'B1': 3, 'B2': 4, 'C1': 5, 'C2': 6}
        self.rows = []
        for row in read_tsv:
            self.rows.append(row)
        
        self.include_salience = include_salience
        self.threshold = threshold
        self.alpha = alpha
        self.beta = beta
        self.salience_threshold = salience_threshold
        self.global_threshold = global_threshold
        self.type_imp = type_imp
        self.level_target = level_target
        self.gamma = gamma
        self.delta = delta
    
    def get_levelled_scores(self):
        levelled_complexities = defaultdict(list)
        not_found, info_not_found = 0, []
        for i, row in enumerate(self.rows):
            curr_sentence = SentenceCEFRLS(row, self.include_salience, self.threshold, self.alpha, self.beta,
                                           self.salience_threshold, self.global_threshold, self.type_imp,
                                           self.level_target, self.gamma, self.delta)
            curr_c = curr_sentence.get_score()
            if curr_c is not None:
                levelled_complexities[curr_sentence.level_word].append(curr_c)
            else:
                not_found +=1
                info_not_found.append((i+1, curr_sentence.complex_word))
        print('Not found: {0}\t{1}'.format(not_found, info_not_found))
        for level, c in levelled_complexities.items():
            print('Level\t {0}\n {1}\t{2}\n{3}'.format(level, np.mean(np.array(c)), len(c), sorted(c)))
    
    def get_compare_synonyms(self):
        for i, row in enumerate(self.rows):
            if i in [1, 2, 6, 7, 8, 9]:
                print('Processing sentence\t {0}'.format(i))
                curr_sentence = SentenceCEFRLS(row, self.include_salience, self.threshold, self.alpha, self.beta,
                                               self.salience_threshold, self.global_threshold, self.type_imp,
                                               self.level_target, self.gamma, self.delta)
                if curr_sentence.word_object:
                    print('Complex word:\t{0}\tLevel:\t{1}'.format(curr_sentence.complex_word, curr_sentence.level_word))
                    annotated_syn = curr_sentence.levelled_syn
                    print('Annotated synonyms: {0}'.format(annotated_syn))
                    print('Annotated synonyms\t\t A1: {0}\t A2: {1}\t B1: {2}\t' 
                        .format(len(annotated_syn['A1']), len(annotated_syn['A2']), len(annotated_syn['B1'])))
                    candidates = curr_sentence.get_synonyms()
                    if len(candidates) > 0:
                        print('Ranked synonyms: {0}'.format(candidates))
                        print('Number of generated synonym candidates: {0}'.format(len(candidates)))
                        scores_positive = [elt for elt in candidates if elt[1] > 0]
                        scores_zero = [elt for elt in candidates if elt[1] == 0]
                        scores_negative = [elt for elt in candidates if elt[1] < 0]
                        print('Number of candidates with score > 0: {0}\t{1}'.format(len(scores_positive), round(100*len(scores_positive)/len(candidates), 2)))
                        print('Number of candidates with score = 0: {0}\t{1}'.format(len(scores_zero), round(100*len(scores_zero)/len(candidates), 2)))
                        print('Number of candidates with score < 0: {0}\t{1}'.format(len(scores_negative), round(100*len(scores_negative)/len(candidates), 2)))
                        top_5 = [elt[0] for elt in candidates[:min(len(candidates), 5)]]
                        print('Final top 5 candidates: {0}'.format(top_5))
                        target_level = annotated_syn[self.nb_to_cefr[self.level_target]]
                        print('# top5 in target level annotations: {0}'.format(len([elt for elt in top_5 if elt in target_level])))
                        store_lower = []
                        for level in range(1, self.level_target + 1):
                            store_lower += annotated_syn[self.nb_to_cefr[level]]
                        print('# top5 in target level + lower annotations: {0}'.format(len([elt for elt in top_5 if elt in store_lower])))
                        store_higher = []
                        for level in range(self.level_target + 1, 4):
                            store_higher += annotated_syn[self.nb_to_cefr[level]]
                        print('# top5 in higher level annotations: {0}'.format(len([elt for elt in top_5 if elt in store_higher])))
                        print('# < 0 in target level annotations: {0}'.format(len([cand for (cand, _) in scores_negative if cand in target_level])))
                        print('# < 0 in target level + lower annotations: {0}'.format(len([cand for (cand, _) in scores_negative if cand in target_level+store_lower])))
                        print('===============\n')
    
    def get_lower_target_higher_level(self, annotated_syn):
        lower, target, higher = [], [], []
        for level in range(1, 4):
            if level < self.level_target:
                lower += annotated_syn[self.nb_to_cefr[level]]
            elif level == self.level_target:
                target += annotated_syn[self.nb_to_cefr[level]]
            else:
                higher += annotated_syn[self.nb_to_cefr[level]]
        return lower, target, higher
    
    def save_numbers(self):
        keys = ['len_a1', 'len_a2', 'len_b1', 'nb_gen_syn', 'nb_gen_syn_pos', 'nb_gen_syn_zero', 'nb_gen_syn_neg',
                'top_5_target_level', 'top_5_target_level_lower', 'top_5_higher_level', "neg_target_level", "neg_target_level_lower"]
        data = {k: [] for k in keys}
        for row in tqdm(self.rows):
            curr_sentence = SentenceCEFRLS(row, self.include_salience, self.threshold, self.alpha, self.beta,
                                           self.salience_threshold, self.global_threshold, self.type_imp,
                                           self.level_target, self.gamma, self.delta)
            if curr_sentence.word_object:
                for (letter, nb) in [('A', '1'), ('A', '2'), ('B', '1')]:
                    data['len_{0}{1}'.format(letter.lower(), nb)].append(len(curr_sentence.levelled_syn['{0}{1}'.format(letter, nb)]))
                candidates = curr_sentence.get_synonyms()
                if len(candidates) > 0:
                    data['nb_gen_syn'].append(len(candidates))
                    scores_negative = [elt for elt in candidates if elt[1] < 0]
                    data['nb_gen_syn_pos'].append(len([elt for elt in candidates if elt[1] > 0]))
                    data['nb_gen_syn_zero'].append(len([elt for elt in candidates if elt[1] == 0]))
                    data['nb_gen_syn_neg'].append(len(scores_negative))

                    top_5 = [elt[0] for elt in candidates[:min(len(candidates), 5)]]
                    lower, target, higher = self.get_lower_target_higher_level(annotated_syn=curr_sentence.levelled_syn)
                    data['top_5_target_level'].append(len([elt for elt in top_5 if elt in target]))
                    data['top_5_target_level_lower'].append(len([elt for elt in top_5 if elt in target+lower]))
                    data['top_5_higher_level'].append(len([elt for elt in top_5 if elt in higher]))
                    data['neg_target_level'].append(len([cand for (cand, _) in scores_negative if cand in target]))
                    data['neg_target_level_lower'].append(len([cand for (cand, _) in scores_negative if cand in target+lower]))
                else:
                    for k in list(set(keys).difference(set(['len_a1', 'len_a2', 'len_b1']))):
                        data[k].append(None)
            else:
                for k in keys:
                    data[k].append(None)
        
        save_rel_path = './final_exp_simplify/cefr_ls_res/{0}_{1}_{2}.pkl'.format(self.level_target, self.gamma, self.delta)
        with open(ROOT_PATH+save_rel_path, 'wb') as outfile:
            pickle.dump(data, outfile)
        return
    
    def save_syn_with_zero_scores(self):
        f_zero_score = open(ROOT_PATH + 'final_exp_simplify/zero_score.txt', 'a+')
        for row in tqdm(self.rows):
            curr_sentence = SentenceCEFRLS(row, self.include_salience, self.threshold, self.alpha, self.beta,
                                           self.salience_threshold, self.global_threshold, self.type_imp,
                                           self.level_target, self.gamma, self.delta)
            if curr_sentence.word_object:
                candidates = curr_sentence.get_synonyms()
                if len(candidates) > 0:
                    scores_zero = [elt[0] for elt in candidates if elt[1] == 0.0]
                    for cand in scores_zero:
                        f_zero_score.write('{0}\n'.format(cand))
        f_zero_score.close()
    
    def save_logs_importance_score(self):
        f_importance_score = open(ROOT_PATH + 'final_exp_simplify/importance_score.txt', 'a+')
        f_importance_score.write('Analysing following parameters\n alpha: {0}\t beta: {1}\n'.format(self.alpha, self.beta))
        levelled_complexities = defaultdict(list)
        not_found, info_not_found = 0, []
        i = 0
        for row in tqdm(self.rows):
            curr_sentence = SentenceCEFRLS(row, self.include_salience, self.threshold, self.alpha, self.beta,
                                           self.salience_threshold, self.global_threshold, self.type_imp,
                                           self.level_target, self.gamma, self.delta)
            curr_c = curr_sentence.get_score()
            if curr_c is not None:
                levelled_complexities[curr_sentence.level_word].append(curr_c)
            else:
                not_found +=1
                info_not_found.append((i+1, curr_sentence.complex_word))
            i += 1
        f_importance_score.write('Not found: {0}\t{1}\n'.format(not_found, info_not_found))
        for level, c in levelled_complexities.items():
            f_importance_score.write('Level\t {0}\n {1}\t{2}\n{3}\n'.format(level, np.mean(np.array(c)), len(c), sorted(c)))
        f_importance_score.write('========================================')
        f_importance_score.close()

    

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-t', '--type', required=True, help="Type of analysis: either `print` or `save`")
    ap.add_argument('-p', '--param', required=True, help="Specific parameter to analyse" +
                                                         "`complexity` for complexity only" +
                                                         "`salience` for salience only" +
                                                         "`cascade` for cascade score, mainly impact of threshold" +
                                                         "`combined` for combined score, mainly impact of threshold" +
                                                         "`selection` for selection score, mainly gamma and delta" +
                                                         "`zero for checking synonyms with zero selection score")
    ap.add_argument('-st', '--salience_threshold', default=0.1, help="Salience threshold to use for cascade experiment")
    ap.add_argument('-a', '--alpha', default=0.5, help="Alpha value to use for combined experiment")
    ap.add_argument('-b', '--beta', default=0.5, help="Beta value to use for combined experiment")
    args = vars(ap.parse_args())

    if args["type"] == "print":
        print("Results for {0} experiment\n ==========".format(args["param"]))
        if args["param"] == "complexity":
            analyzer = SentencesCEFRLS()
            analyzer.get_levelled_scores()

        if args["param"] == "salience":
            analyzer = SentencesCEFRLS(include_salience=True, alpha=0, beta=1,
                                       global_threshold=0, type_imp="combined")
            analyzer.get_levelled_scores()
        
        if args["param"] == "cascade":
            analyzer = SentencesCEFRLS(include_salience=True, type_imp="cascade",
                                   salience_threshold=args["salience_threshold"])
            analyzer.get_levelled_scores()
        
        if args["param"] == "combined":
            analyzer = SentencesCEFRLS(include_salience=True, alpha=args["alpha"],
                                   beta=args["beta"], type_imp="combined")
            analyzer.get_levelled_scores()
    
    if args["type"] == "save":
        if args["param"] == "selection":
            print("Results for generated synonyms \n ==========")

            def run_store_results(level, gamma, delta):
                analyzer = SentencesCEFRLS(level_target=level, gamma=gamma, delta=delta)
                analyzer.save_numbers()

            params = [(2, 0.5, 0.5), (3, 0.5, 0.5), (2, 0.7, 0.3), (3, 0.7, 0.3), (2, 0.3, 0.7), (3, 0.3, 0.7),
                      (2, 0.6, 0.4), (3, 0.6, 0.4), (2, 0.4, 0.6), (3, 0.4, 0.6), (2, 0.8, 0.2), (3, 0.8, 0.2),
                      (2, 0.2, 0.8), (3, 0.2, 0.8)]
            for (level, gamma, delta) in params:
                run_store_results(level, gamma, delta)
        
        if args["param"] == "zero":
            print('Saving synonyms with zero scores')
            analyzer = SentencesCEFRLS(level_target=3, gamma=0.5, delta=0.5)
            analyzer.save_syn_with_zero_scores()
        
        if args["param"] == "combined":
            print('Saving results for combined scores')
            params = [(0.2, 0.8), (0.3, 0.7), (0.4, 0.6), (0.5, 0.5), (0.6, 0.4), (0.7, 0.3), (0.8, 0.2)]
            for (alpha, beta) in params:
                test = SentencesCEFRLS(alpha=alpha, beta=beta, type_imp="combined",
                                       include_salience=True, global_threshold=0)
                test.save_logs_importance_score()

