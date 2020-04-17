#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 16:37:26 2019

@author: siangooding
"""

import lexical_simplification.labeler as labeler
import lexical_simplification.experiment as experiment
import collections
import pandas as pd 

class Complexity_labeller:

	def __init__(self, model_path, temp_file):

		self.model_path = model_path
		self.temp_file = temp_file

	def initialise_model(self):
		# model = labeler.SequenceLabeler.load(self.model_path)
		self.model = labeler.SequenceLabeler.load(self.model_path)
		# config = model.config
		self.config = self.model.config
		# predictions_cache = {}

		id2label = collections.OrderedDict()
		# for label in model.label2id:
		# 	id2label[model.label2id[label]] = label
		for label in self.model.label2id:
			id2label[self.model.label2id[label]] = label

	def convert_format_string(self, string):
	    from nltk import word_tokenize
	    split_list = word_tokenize(string)
	    
	    dataframe = pd.DataFrame()
	    dataframe['word'] = split_list
	    dataframe['binary'] = 'N'
	    dataframe.to_csv(self.temp_file, sep = '\t',index=False, header=False, quotechar=' ') 
	    
	def convert_format_token(self, tokenised_string):
	    dataframe = pd.DataFrame()
	    dataframe['word'] = tokenised_string
	    dataframe['binary'] = 'N'
	   
	    dataframe.to_csv(self.temp_file, sep = '\t',index=False, header=False, quotechar=' ') 
	    
	    
	def convert_rasp_table(self, rasp_table):
	    words_ = rasp_table.word.values
	    dataframe = pd.DataFrame()
	    dataframe['word'] = words_
	  
	    dataframe['binary'] = 'N'
	    dataframe.to_csv(self.temp_file, sep = '\t',index=False, header=False, quotechar=' ') 
	    
	def get_dataframe(self):
	    
		sentences_test = experiment.read_input_files(self.temp_file)
	    # batches_of_sentence_ids = experiment.create_batches_of_sentence_ids(sentences_test, config["batch_equal_size"], config['max_batch_size'])
		batches_of_sentence_ids = experiment.create_batches_of_sentence_ids(sentences_test, self.config["batch_equal_size"], self.config['max_batch_size'])
	    
		for sentence_ids_in_batch in batches_of_sentence_ids:
			batch = [sentences_test[i] for i in sentence_ids_in_batch]
	        # cost, predicted_labels, predicted_probs = model.process_batch(batch, is_training=False, learningrate=0.0)
			_, predicted_labels, predicted_probs = self.model.process_batch(batch, is_training=False, learningrate=0.0)
		try:
			assert(len(sentence_ids_in_batch) == len(predicted_labels))
		except:
			print('batch size error')
	    
	    
		prob_labels = predicted_probs[0]
		probability_list = []
		for prob_pair in prob_labels:
			probability_list.append(prob_pair[1])
	    
		annotated_sentences = pd.DataFrame()
	    
		sentences =[sentences_test[i] for i in sentence_ids_in_batch]
	    
		annotated_sentences['index'] = sentence_ids_in_batch
	    
		annotated_sentences['sentences'] = sentences
	    
		annotated_sentences['labels'] = predicted_labels
	    
		annotated_sentences['probs'] = predicted_probs
	    
	    
		return annotated_sentences

	def get_bin_labels(self):
	    
	    # sentences_test = experiment.read_input_files(temp_file)
		sentences_test = experiment.read_input_files(self.temp_file)
	    # batches_of_sentence_ids = experiment.create_batches_of_sentence_ids(sentences_test, config["batch_equal_size"], config['max_batch_size'])
		batches_of_sentence_ids = experiment.create_batches_of_sentence_ids(sentences_test, self.config["batch_equal_size"], 
																			self.config['max_batch_size'])
	    
		for sentence_ids_in_batch in batches_of_sentence_ids:
			batch = [sentences_test[i] for i in sentence_ids_in_batch]
	        # cost, predicted_labels, predicted_probs = model.process_batch(batch, is_training=False, learningrate=0.0)
			_, predicted_labels, predicted_probs = self.model.process_batch(batch, is_training=False, learningrate=0.0)
	    
		try:
			assert(len(sentence_ids_in_batch) == len(predicted_labels))
		except:
			print('cw error')
	    
		prob_labels = predicted_probs[0]
		probability_list = []
		for prob_pair in prob_labels:
			probability_list.append(prob_pair[1])

		return predicted_labels

	def get_prob_labels(self):
	  
	    try:
	        sentences_test = experiment.read_input_files(temp_file)
	   
	        batches_of_sentence_ids = experiment.create_batches_of_sentence_ids(sentences_test, config["batch_equal_size"], config['max_batch_size'])
	    except:
	        return 'error'
	    for sentence_ids_in_batch in batches_of_sentence_ids:
	        batch = [sentences_test[i] for i in sentence_ids_in_batch]
			# cost, predicted_labels, predicted_probs = model.process_batch(batch, is_training=False, learningrate=0.0)
	        _, predicted_labels, predicted_probs = self.model.process_batch(batch, is_training=False, learningrate=0.0)
	    
	    try:
	        assert(len(sentence_ids_in_batch) == len(predicted_labels))
	    except:
	        return 'error'
	    prob_labels = predicted_probs[0]
	    probability_list = []
	    for prob_pair in prob_labels:
	        probability_list.append(prob_pair[0])

	    return probability_list