# -*- coding: utf-8 -*-py
import yaml
import stanfordnlp
from stanfordnlp_.stanfordnlp_parser import config_stanford_nlp
from lexical_simplification.simplification import Sentence, Word
from lexical_simplification.scores import SelectionScore, ImportanceSimplifyScore

NLP = stanfordnlp.Pipeline(**config_stanford_nlp['pre_tokenized'])

class RecursiveSimplification:
    def __init__(self, recursive_config_path, nlp=NLP):
        """ Recursive Simplification system, with configuration file path as main parameters.
        The .yaml config file should contain the followings : 

        Global parameters:
        `cached_syn_path`: .yaml path to cached synonyms, by default and assuming root directory,
        should be `./lexical_simplification/cached_synonyms.yaml`
        `f_save`: path to store logs for simplification. If not used should be set to None.
        
        Parameters for complexity + salience:
        `include_salience`: boolean, to incorporate salience or not.
        If set to None system unchanged from original.
        `threshold`: complexity threshold (only words with a complexity superior to it 
        will be simplified). Should always be non null.
        The below parameters are useful only if `include_salience` is True.
        Else set each key to None.
        `type_imp`: type of score to use for using both complexity and salience. 
        Should be either `cascade` or `combined`
        `alpha` and `beta`: coefficients if `type_imp` is set to `combined`
        `global_threshold`: selection for simplification threshold
        if `type_imp` is set to `combined`
        `salience_threshold`: salience threshold if `type_imp` is `cascade`
        
        Parameters for pre-selecting substitution candidates
        `level_target`: target level for simplification. 
        Either integer between 1 and 6, or None. If set to None system unchanged from original.
        `gamma` and `delta`: coefficients if `level_target` is not None """
        with open(recursive_config_path) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        
        self.cached_syn_path = config["cached_syn_path"]
        self.f_save = config["f_save"]

        self.include_salience = config["include_salience"]
        self.threshold = config["threshold"]
        self.type_imp = config["type_imp"]
        self.alpha = config["alpha"]
        self.beta = config["beta"]
        self.global_threshold = config["global_threshold"]
        self.salience_threshold = config["salience_threshold"]

        self.level_target = config["level_target"]
        self.gamma = config["gamma"]
        self.delta = config["delta"]

        self.init_scores()
        self.nlp = nlp
    
    def init_scores(self):
        self.importance_score = ImportanceSimplifyScore(include_salience=self.include_salience,
                                                        alpha=self.alpha, beta=self.beta,
                                                        threshold=self.threshold, salience_threshold=self.salience_threshold,
                                                        global_threshold=self.global_threshold, type_imp=self.type_imp)
        if self.level_target is not None:
            self.selection_score = SelectionScore(level_target=self.level_target,
                                                  gamma=self.gamma, delta=self.delta)
    
    def write_log(self, type_log, unit_object):
        if self.f_save is not None:
            if type_log == 'complex':
                self.f_save.write('Tokenized sentence \n')
                self.f_save.write('{0} \n'.format(unit_object.tokenized))
                self.f_save.write('Complex words \n')
                self.f_save.write('{0} \n'.format(unit_object.complex_words))
            
            if type_log == 'most_complex':
                self.f_save.write('Most complex word \n')
                self.f_save.write('{0} {1} \n'.format(unit_object.word, unit_object.pos))
            
            if type_log == 'synonym':
                self.f_save.write('Synonyms \n')
                self.f_save.write('{0} \n'.format(unit_object.synonyms))
                self.f_save.write('=============== \n')
    
    def apply_simplification(self, tokenized_text, ignore_list=[]):
        sentence_object = Sentence(tokenized_text, self.threshold, ignore_list, self.nlp, self.importance_score)
        print('tokenized sentence: {0}'.format(sentence_object.tokenized))
        print('complex words: {0}'.format(sentence_object.complex_words))
        self.write_log(type_log='complex', unit_object=sentence_object)

        if (len(sentence_object.complex_words) > 0):  # At least one word to simplify
            
            # Word object from the most important word to simplify
            (index,_), *_ = sentence_object.complex_words
            word_object = Word(sentence_object, index, self.cached_syn_path, self.nlp)

            print('Most important word to simplify\t {0}\t {1}\t'.format(word_object.word, word_object.pos))
            self.write_log(type_log='most_complex', unit_object=word_object)
        
            # Synonym generation + selection + ranking
            word_object.get_synonyms()  # generation
            if self.level_target is not None:  # selection
                word_object.select_synonyms(selection_score=self.selection_score)  
            print('synonyms:\t {0}'.format(word_object.synonyms))
            self.write_log(type_log='synonym', unit_object=word_object)
            print('===\n \n')
            synonym = word_object.get_ranked_synonyms()  # ranking

            if synonym != []:
                sentence_object.make_simplification(synonym, word_object.index)

            return self.apply_simplification(sentence_object.tokenized, sentence_object.ignore_index)

        else:
            return sentence_object.tokenized


if __name__ == '__main__':
    RECURSIVE_CONFIG_PATH = '/home/ib431/Documents/projects/cam_mphil_project/lexical_simplification/recursive_config.yaml'
    recursive_simplification = RecursiveSimplification(recursive_config_path=RECURSIVE_CONFIG_PATH)

    import nltk
    SENTENCE_1 = nltk.word_tokenize("Sustainability is a big issue in the contemporary world.")
    # ['Sustainability', 'is', 'a', 'big', 'issue', 'in', 'the', 'modern', 'world']
    SENTENCE_2 = nltk.word_tokenize("Motorists are being urged to take extra care in treacherous conditions.")
    # ['Motorists', 'are', 'being', 'advised', 'to', 'take', 'extra', 'care', 'in', 'dangerous', 'conditions', '.']
    simplified = recursive_simplification.apply_simplification(tokenized_text=SENTENCE_1)
    print(simplified)