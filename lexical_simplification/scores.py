# -*- coding: utf-8 -*-py
import csv
import pickle
from summa.keywords import keywords
import lexical_simplification.complex_word as complex_word
from lexical_simplification.traditional_features import get_key_val, is_string_of_float
from lexical_simplification.traditional_features import TO_DISCARD
from settings.settings import ROOT_PATH 

class SelectionScore:
    def __init__(self, level_target, gamma, delta,
                 tsv_paths=[ROOT_PATH + 'readability_code/resources/CEFR_levels.tsv',
                            ROOT_PATH + 'readability_code/resources/KET_list.tsv',
                            ROOT_PATH + 'readability_code/resources/PET_list.tsv'],
                 freq_info_path=ROOT_PATH + 'cefr_freq_info.pkl',
                 to_discard=TO_DISCARD):
        self.nb_level_to_written = {2: 'KET', 3: 'PET', 4: 'FCE', 5: 'CAE', 6: 'CPE'}
        self.level_target = level_target
        self.gamma = gamma
        self.delta = delta

        self.to_discard = to_discard
        self.dict_levels = self.build_dict_from_tsv(tsv_paths=tsv_paths)
        self.freq_info = pickle.load(open(freq_info_path, 'rb'))
    
    def build_dict_from_tsv(self, tsv_paths):
        """ From .tsv files containing levelled words, build a dictionary
        Key = word, value = level """
        res = {}
        for tsv_path in tsv_paths:
            with open(tsv_path) as tsvfile:
                reader = csv.DictReader(tsvfile, dialect='excel-tab')
                for row in reader:
                    word, level = get_key_val(row=row, tsv_path=tsv_path)
                    if ("CEFR_levels" not in tsv_path) and (not is_string_of_float(s=word)):
                        # KET and PET files only, if not number
                        res[word] = level
                    if ("CEFR_levels" in tsv_path) and ('-' not in word):
                        # CEFR only, expressions linked with `-` are also separated by ' ' in the .tsv
                        res[word] = level

                    sub_words = word.split(" ")
                    if len(sub_words) > 1:
                        for sub_word in sub_words:
                            if sub_word == "sth":
                                res["something"] = level
                            if sub_word not in self.to_discard:
                                res[sub_word] = level
        return res
    
    def get_word_level(self, word):
        if word in self.dict_levels.keys():
            return int(self.dict_levels[word])
        return 0  # 0 corresponds to non found level
    
    def get_cefr_score(self, word):
        level_word = self.get_word_level(word=word)
        if level_word == 0:
            return 0
        elif level_word  == self.level_target:
            return 1
        else:
            return 0.2 * (self.level_target - level_word)

    def get_freq_score(self, word):
        if self.level_target == 1:
            return 0
        
        level_target = self.nb_level_to_written[self.level_target]
        if self.freq_info["tot_count"][level_target] == 0:
            return 0
        self.freq_info["word_count"][level_target].setdefault(word, 0)
        return self.freq_info["word_count"][level_target][word] / self.freq_info["tot_count"][level_target]
    
    def get_pre_selection_score(self, word):
        return self.gamma * self.get_cefr_score(word) + self.delta * self.get_freq_score(word)


class ImportanceSimplifyScore:
    def __init__(self, include_salience, alpha=None, beta=None,
                 threshold=0.5, salience_threshold=0.2, global_threshold=0.5, type_imp='combined'):
        """ This score determines which words from the original sentences should be simplified
        @param include_salience: boolean. If False only using the encoder-decoder like in the original version
        @param alpha: if using `combined` option, multiplicator for complexity scores
        @param beta: if using `combined` option, multiplicator for salience scores
        @param threshold: if not using salience, filter for complexity words
        @param threshold: if using `cascade` option, first filter for salience words
        @param global_threshold: if using `combined` option, filter for final words
        @param type_imp: type of method to select final words. 
        If `cascade`, first selecting most salient words according to `salience_threshold filter`,
        then computing the complexity for each salient word
        and filtering those with complexity above `threshold`
        If `combined`, filtering words with final scores above `global_threshold`, 
        with final_score = alpha * complexity_score + beta * salience_score """
        self.include_salience = include_salience
        self.alpha = alpha
        self.beta = beta

        self.threshold = threshold  # filter for complexity only (no use of salience)
        self.salience_threshold = salience_threshold  # filter for salience only when cascade
        self.global_threshold = global_threshold  # filter for combined

        self.type_imp = type_imp

    def get_complexity_scores(self, ignore_index, tokenized, indexes, threshold):
        complex_words = [(a,b) for a,b in list(zip([a for a,b in indexes], complex_word.get_complex_words(tokenized))) if b > threshold]
        complex_words = [(a,b) for a,b in complex_words if a not in ignore_index]
        return sorted(complex_words, key = lambda x: x[1], reverse=True)
    
    def get_salience_scores(self, tokenized):
        # Computing salience scores with TextRank algorithms
        # Implementation taken from https://github.com/summanlp/textrank and modified
        return keywords(' '.join(tokenized), ratio=1, scores=True, return_single_words_score=True)
    
    def get_combined_scores(self, ignore_index, tokenized, indexes):
        complex_words = self.get_complexity_scores(ignore_index, tokenized, indexes, threshold=0)
        complex_words = [(a, self.alpha * b) for (a, b) in complex_words]

        salience_scores = self.get_salience_scores(tokenized)
        final_words = []
        
        for (index, score) in complex_words:
            if tokenized[index] in salience_scores.keys():
                final_words.append((index, score + self.beta * salience_scores[tokenized[index]]))
            else:
                final_words.append((index, score))

        final_words = [(index, score) for (index, score) in final_words if score > self.global_threshold]
        return sorted(final_words, key = lambda x: x[1], reverse=True)
    
    def get_cascade_scores(self, ignore_index, tokenized, indexes):
        salience_scores = self.get_salience_scores(tokenized)
        non_salient_words = [word for word, score in salience_scores.items() if score <= self.salience_threshold]
        add_ignore_index = [i for i in range(len(tokenized)) if tokenized[i] in non_salient_words]
        return self.get_complexity_scores(ignore_index=ignore_index+add_ignore_index, tokenized=tokenized, indexes=indexes, threshold=self.threshold)
    
    def get_scores(self, ignore_index, tokenized, indexes):
        if not self.include_salience:
            return self.get_complexity_scores(ignore_index, tokenized, indexes, threshold=self.threshold)
        elif self.type_imp == "combined":
            return self.get_combined_scores(ignore_index, tokenized, indexes)
        else:  # cascade 
            return self.get_cascade_scores(ignore_index, tokenized, indexes)
        
        
if __name__ == '__main__':
    # selection_score = SelectionScore(level_target=5, gamma=0.5, delta=0.5)
    # word = "careznvnzen"
    # cefr_score = selection_score.get_cefr_score(word=word)
    # freq_score = selection_score.get_freq_score(word=word)
    # print(cefr_score, freq_score)

    import nltk
    importance_simplify_score = ImportanceSimplifyScore(alpha=0.8, beta=0.2, include_salience=False)
    ignore_index = []
    threshold = 0.5
    tokenized = nltk.word_tokenize("Sustainability is a big issue in the contemporary world. Motorists are being urged to take extra care in treacherous conditions.")
    indexes = list(enumerate(tokenized))
    final_words = importance_simplify_score.get_scores(ignore_index, tokenized, indexes)
    print(final_words)