# -*- coding: utf-8 -*-py
import yaml
import stanfordnlp
import gensim.downloader as api
from allennlp.commands.elmo import ElmoEmbedder
from stanfordnlp_.stanfordnlp_parser import config_stanford_nlp
from lexical_simplification.simplification import Sentence, Word
from lexical_simplification.scores import SelectionScore, ImportanceSimplifyScore

NLP = stanfordnlp.Pipeline(**config_stanford_nlp['pre_tokenized'])
ELMO = ElmoEmbedder()
GLOVE_MODEL = api.load('glove-wiki-gigaword-50')
WORD2VEC_MODEL = api.load('word2vec-google-news-300')

class RecursiveSimplification:
    def __init__(self, recursive_config_path, nlp=NLP, elmo=ELMO,
                 glove_model=GLOVE_MODEL, word2vec_model=WORD2VEC_MODEL):
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
        self.elmo = elmo
        self.glove_model = glove_model
        self.word2vec_model = word2vec_model

        self.param_to_key = {'include_salience': self.include_salience, 'threshold': self.threshold,
                             'type_imp': self.type_imp, 'alpha': self.alpha, 'beta': self.beta,
                             'global_threshold': self.global_threshold, 'salience_threshold': self.salience_threshold,
                             'level_target': self.level_target, 'gamma': self.gamma, 'delta': self.delta,
                             'f_save': self.f_save}
    
    def init_scores(self):
        self.importance_score = ImportanceSimplifyScore(include_salience=self.include_salience,
                                                        alpha=self.alpha, beta=self.beta,
                                                        threshold=self.threshold, salience_threshold=self.salience_threshold,
                                                        global_threshold=self.global_threshold, type_imp=self.type_imp)
        if self.level_target is not None:
            self.selection_score = SelectionScore(level_target=self.level_target,
                                                  gamma=self.gamma, delta=self.delta)
    
    def set_param(self, param, value):
        if param in self.param_to_key:
            if param == 'f_save':
                self.f_save = value
            else:
                self.param_to_key[param] = value
                self.init_scores()
    
    def write_log(self, type_log, unit_object):
        if self.f_save is not None:
            save_log = open(self.f_save, 'a')

            if type_log == 'complex':
                save_log.write('Tokenized sentence \n')
                save_log.write('{0} \n'.format(unit_object.tokenized))
                save_log.write('Complex words \n')
                save_log.write('{0} \n'.format(unit_object.complex_words))
            
            if type_log == 'most_complex':
                save_log.write('Most complex word \n')
                save_log.write('{0} {1} \n'.format(unit_object.word, unit_object.pos))
            
            if type_log == 'synonym':
                save_log.write('Synonyms \n')
                save_log.write('{0} \n'.format(unit_object.synonyms))
                save_log.write('=============== \n')
            
            save_log.close()
    
    def apply_simplification(self, tokenized_text, ignore_list=[]):
        sentence_object = Sentence(tokenized_text, self.threshold, ignore_list, self.nlp, self.importance_score)
        print('tokenized sentence: {0}'.format(sentence_object.tokenized))
        print('complex words: {0}'.format(sentence_object.complex_words))
        self.write_log(type_log='complex', unit_object=sentence_object)

        if (len(sentence_object.complex_words) > 0):  # At least one word to simplify
            
            # Word object from the most important word to simplify
            (index,_), *_ = sentence_object.complex_words
            word_object = Word(sentence_object, index, self.cached_syn_path, self.nlp, self.elmo,
                               self.glove_model, self.word2vec_model)

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
    # SENTENCE_1 = nltk.word_tokenize("Sustainability is a big issue in the contemporary world.")
    # # ['Sustainability', 'is', 'a', 'big', 'issue', 'in', 'the', 'modern', 'world']
    # SENTENCE_2 = nltk.word_tokenize("Motorists are being urged to take extra care in treacherous conditions.")
    # # ['Motorists', 'are', 'being', 'advised', 'to', 'take', 'extra', 'care', 'in', 'dangerous', 'conditions', '.']
    # simplified_1 = recursive_simplification.apply_simplification(tokenized_text=SENTENCE_1)
    # print(simplified_1)
    # simplified_2 = recursive_simplification.apply_simplification(tokenized_text=SENTENCE_2)
    # print(simplified_2)
    SENTENCE_3 = nltk.word_tokenize("South American Indians have chewed coca leaves for centuries. The leaves reputedly provide energy and are said to have medicinal qualities. Supporters of Bolivia’s position praised it for standing up for the rights of indigenous people. “The Bolivian move is inspirational and groundbreaking,” said Danny Kushlick, Head of External Affairs at the Transform Drug Policy Foundation, which promotes drug liberalization. “It shows that any country that has had enough of the war on drugs can change the terms of its engagement with the UN conventions.” ")
    simplified_3 = recursive_simplification.apply_simplification(tokenized_text=SENTENCE_3)
    print(simplified_3)

