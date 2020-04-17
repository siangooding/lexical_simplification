# -*- coding: utf-8 -*-
import yaml
import nltk
import lexical_simplification.complex_word as complex_word
import spacy
import json
import urllib
import lexical_simplification.plural as plural 
import lexical_simplification.verb as verb
import pandas as pd
import lexical_simplification.helper_functions as hf
import string
import inflect


class Sentence:

	def __init__(self, tokenized, threshold, ignore_list, nlp,
				 importance_score):
		self.threshold = threshold
		self.tokenized = tokenized
		self.indexes = list(enumerate(self.tokenized))
		self.pos_tags = nltk.pos_tag(self.tokenized)
		if ignore_list == []:
			self.ignore_index = [c for (a,b),(c,d) in zip(self.pos_tags, self.indexes) if 'P' in b]
		else:
			self.ignore_index = ignore_list
		self.nlp = nlp

		self.importance_score = importance_score
		self.complex_words = \
			importance_score.get_scores(self.ignore_index, self.tokenized, self.indexes)
	
	def add_ignore(self, item):
		self.ignore_index.append(item)
	
	def make_simplification(self, synonym, index):

		tokens = self.tokenized
		del tokens[index]

		for i,word in enumerate(synonym):
			tokens.insert((index + i), word)
			self.add_ignore(index)
			
		self.tokenized = tokens
		self.indexes = list(enumerate(self.tokenized))
		self.pos_tags = nltk.pos_tag(self.tokenized)

		self.complex_words = \
			self.importance_score.get_scores(self.ignore_index, self.tokenized, self.indexes)


class Word:

	def __init__(self, sentence_object, index, cached_syn_path, nlp):

		pos_tags = sentence_object.pos_tags
		self.token_sent = sentence_object.tokenized
		self.pos_sent = nltk.pos_tag(self.token_sent)
		self.word = pos_tags[index][0]
		self.pos = pos_tags[index][1]
		self.index = index
		self.synonyms = []
		self.tense = None
		self.lemma = None
		self.is_plural = True if (self.pos == 'NNS') else False
		self.nlp = nlp
 
		# To cache the synonyms generated to speed up the process in the long term
		self.cached_syn_path = cached_syn_path
		self.cached_synonyms = self.load_cached_synonyms(cached_syn_path)
		if self.cached_synonyms is None:  # no synonym cached at this point
			self.cached_synonyms = {}

		# Mapping resource to generate synonyms to corresponding function
		self.get_synonym_from_resource = {'moby_thesaurus': self.get_synonym_moby_thesaurus,
										  'wordnet': self.get_synonym_wordnet,
										  'glove_embeddings': self.get_synonym_embedding,
										  'word2vec_embeddings': self.get_synonym_embedding}
		self.get_synonym_from_resource = {'glove_embeddings': self.get_synonym_embedding,
										  'word2vec_embeddings': self.get_synonym_embedding}
		self.model_embedding = {'glove_embeddings': 'glove-wiki-gigaword-50',
								'word2vec_embeddings': 'word2vec-google-news-300'}
	

	def load_cached_synonyms(self, cached_syn_path):
		with open(cached_syn_path) as yaml_file:
			cached_synonyms = yaml.load(yaml_file, Loader=yaml.FullLoader)
		return cached_synonyms


	def update_cached_synonyms(self, lemma, resource, all_synonyms):
		if lemma not in self.cached_synonyms.keys():
			self.cached_synonyms[lemma] = {}
		self.cached_synonyms[lemma][resource] = all_synonyms

		with open(self.cached_syn_path, "w") as f:
			yaml.dump(self.cached_synonyms, f)
		
		self.cached_synonyms = self.load_cached_synonyms(self.cached_syn_path)
	

	def get_cached_synonym_by_resource(self, lemma, resource):
		if (lemma in self.cached_synonyms.keys()) and \
			(resource in self.cached_synonyms[lemma].keys()) :  # Checking if lemma's synonyms cached already
			all_synonyms = self.cached_synonyms[lemma][resource]
		else:
			all_synonyms = False
		return all_synonyms
	

	def filter_synonyms_by_pos(self, synonyms, filter_pos):
		if len(filter_pos) > 0:
			return(list(set([syn[0] for syn in synonyms if syn[1] in filter_pos])))
		else:
			return(list(set([syn[0] for syn in synonyms])))


	def get_synonym_moby_thesaurus(self, lemma, resource): 
		found = False

		# First, getting all possible synonyms, regardless of the POS tag
		all_synonyms = self.get_cached_synonym_by_resource(lemma, resource)
		if not all_synonyms:  # API call to get synonyms from Moby Thesaurus [http://moby-thesaurus.org/]
			import requests
			r = requests.get(url='http://words.bighugelabs.com/api/2/d8e9e06ab1208c4a35dc91b16f4f42a3/{0}/json'
								.format(lemma))
			if r.status_code == 500:
				print('GET API failed, may be due to exceeded number of requests')
				print('Unable to find synonyms for {0} on Moby Thesaurus'.format(lemma))
			elif r.status_code == 200:  # API call succeeded 
				found = True
				all_synonyms = r.json()
				self.update_cached_synonyms(self.lemma, resource, all_synonyms)
		else:
			found = True

		# Second, if synonyms found, filtering depending on POS tag
		if found:
			try:
				if 'V' in self.pos:
					self.synonyms += all_synonyms["verb"]["syn"]
					try:
						self.tense = verb.verb_tense(self.word)
					except Exception as e:
						print(e)
						pass
				elif 'N' in self.pos:
					try:
						self.synonyms += all_synonyms["noun"]["syn"] + all_synonyms["noun"]["sim"]
					except:
						self.synonyms += all_synonyms["noun"]["syn"]
				elif 'J' in self.pos:
					try:
						self.synonyms += all_synonyms["adjective"]["syn"] + all_synonyms["adjective"]["sim"]
					except:
						self.synonyms += all_synonyms["adjective"]["syn"]
				elif 'RB' in self.pos:
					try:
						self.synonyms += all_synonyms["adverb"]["syn"] + all_synonyms["adverb"]["sim"]
					except:
						self.synonyms += all_synonyms["adverb"]["syn"]
			except:
				total_list = []
				for pos in all_synonyms:
					for type_ in all_synonyms[pos]:
						total_list.append(all_synonyms[pos][type_])
				# taking everything as potential synonyms 
				self.synonyms += [item for sublist in total_list for item in sublist]
	

	def get_synonym_wordnet(self, lemma, resource):
		# First, getting all possible synonyms, regardless of the POS tag
		all_synonyms = self.get_cached_synonym_by_resource(lemma, resource)
		if not all_synonyms:
			from nltk.corpus import wordnet as wn
			all_synonyms = wn.synsets(lemma)
			all_synonyms = [elt.name() for elt in all_synonyms]
			to_keep = []
			for elt in all_synonyms:
				if elt.split('.')[0] != lemma:
					to_keep.append(elt)
			self.update_cached_synonyms(self.lemma, resource, to_keep)
		

		# Second, filtering on POS tag
		helper = [synset.split('.') for synset in all_synonyms]
			
		if 'V' in self.pos:
			self.synonyms += self.filter_synonyms_by_pos(synonyms=helper, 
														 filter_pos=['v'])
			try:
				self.tense = verb.verb_tense(self.word)
			except Exception as e:
				print(e)
				pass
		elif 'N' in self.pos:
			self.synonyms += self.filter_synonyms_by_pos(synonyms=helper, 
														 filter_pos=['n'])
		elif 'J' in self.pos:
			self.synonyms += self.filter_synonyms_by_pos(synonyms=helper, 
														 filter_pos=['a', 's'])
		elif 'RB' in self.pos:
			self.synonyms += self.filter_synonyms_by_pos(synonyms=helper, 
														 filter_pos=['r'])
		else:
			self.synonyms += self.filter_synonyms_by_pos(synonyms=helper, 
														 filter_pos=[])

	def sort_synonym_embedding(self, lemma, synonyms, nlp, max_nb=10):
		found = 0
		index = 0
		selected = []
		while (found < max_nb) and (index < len(synonyms)):
			candidate = synonyms[index][0]
			if lemma not in candidate:
				# Avoiding cases where the lemma is `species` 
				# and the candidate `fish species` for example
				# not simplifying as the complex word is still here
				lemma_stanford = nlp([[lemma]]).sentences[0].words[0]
				candidate_stanford = nlp([[candidate]]).sentences[0].words[0]
				if lemma_stanford.lemma != candidate_stanford.lemma:
					selected.append([candidate_stanford.lemma, candidate_stanford.upos])
					found += 1
			index += 1
		return selected

	def get_synonym_embedding(self, lemma, resource):
		all_synonyms = self.get_cached_synonym_by_resource(lemma, resource)
		if not all_synonyms:
			import gensim.downloader as api
			model = api.load(self.model_embedding[resource])
			try:
				all_synonyms = model.most_similar(lemma, topn=30)
				# all_synonyms = [list(nltk.pos_tag([elt[0]])[0]) for elt in all_synonyms]
				all_synonyms = [[elt[0].replace('_', ' '), elt[1]] for elt in all_synonyms]
				self.update_cached_synonyms(self.lemma, resource, all_synonyms)
			except Exception as e:  # word not in vocabulary
				print(e)
				pass
		
		# Second, filtering on POS tag
		
		if all_synonyms:
			all_synonyms = self.sort_synonym_embedding(lemma, all_synonyms, self.nlp)
			if 'V' in self.pos:
				self.synonyms += self.filter_synonyms_by_pos(synonyms=all_synonyms, 
															filter_pos=['VERB'])
				try:
					self.tense = verb.verb_tense(self.word)
				except Exception as e:
					print(e)
					pass
			elif 'N' in self.pos:
				self.synonyms += self.filter_synonyms_by_pos(synonyms=all_synonyms, 
															filter_pos=['NOUN'])
			elif 'J' in self.pos:
				self.synonyms += self.filter_synonyms_by_pos(synonyms=all_synonyms, 
															filter_pos=['ADJ'])
			elif 'RB' in self.pos:
				self.synonyms += self.filter_synonyms_by_pos(synonyms=all_synonyms, 
															filter_pos=['ADV'])
			else:
				self.synonyms += self.filter_synonyms_by_pos(synonyms=all_synonyms, 
															filter_pos=[])


	def get_synonyms(self):
		
		spacy_module = spacy.load('en')

		doc = spacy_module("".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in self.token_sent]).strip())

		for token in doc:
			if str(token) == self.word:
				self.lemma = token.lemma_
		
		if (self.lemma is not None) and (self.word not in ['', '-', '”', ':']):  # word's lemma correctly identified ==> searching for synonyms
			# Updating list of synonyms with all possible resources
			for resource_type, f_to_synonym in self.get_synonym_from_resource.items():
				f_to_synonym(lemma=self.lemma, resource=resource_type)

		
		self.synonyms = [x.split(' ') for x in self.synonyms]

		temp_set = []

		for word in self.synonyms:
			temp_set.append(word[0])

		temp_set = set(temp_set)

		temp_set = [[x] for x in temp_set]

		self.synonyms = temp_set


		if self.is_plural == True:
			p = inflect.engine()
			all_synonyms = []
			for synonym in self.synonyms:
				new_synonyms = []
				for word in synonym:
					if p.singular_noun(word) == word:
						new_synonyms.append(plural.noun_plural(word))
					else:
						new_synonyms.append(word)
				all_synonyms.append(new_synonyms)
					

			self.synonyms = all_synonyms

		if self.tense != None:
			tense_synonyms = []
			for x in self.synonyms:
				multi_word = []
				for element in x:
					try:
						multi_word.append((verb.verb_conjugate(element, tense=self.tense, negate=False)))
					except:
						multi_word.append(element)
				tense_synonyms.append(multi_word)
				
			self.synonyms = tense_synonyms
		
		self.synonyms = [[[syn for syn in syn_l if syn not in ['', None]] 
							for syn_l in self.synonyms]]
		self.synonyms = [syn_l for sub_list in self.synonyms if sub_list != [] for syn_l in sub_list]

	
	def select_synonyms(self, selection_score, topn=5):
		if len(self.synonyms) > topn:
			self.synonyms = [syn[0] for syn in self.synonyms]
			scored = [(word, selection_score.get_pre_selection_score(word)) for word in self.synonyms]
			scored = sorted(scored, key=lambda tup: tup[1], reverse=True)
			print(scored)
			self.synonyms = [[syn[0]] for syn in scored[:topn]]
		return
			

	def get_ranked_synonyms(self):
		if len(self.synonyms) > 1:
			synonym_scores = pd.DataFrame()
			#print("synonyms", self.synonyms)
			synonym_scores['synonyms'] = self.synonyms
			#print("elmo", hf.get_elmo_score(self.synonyms, self.token_sent, self.index))
			synonym_scores['sem_sim'] = hf.get_elmo_score(self.synonyms, self.token_sent, self.index)
			#print("complexity", complex_word.get_synonym_complexities(self.synonyms, self.token_sent, self.index))
			synonym_scores['complexity'] = complex_word.get_synonym_complexities(self.synonyms, self.token_sent, self.index)
			#print("grammaticality", hf.get_gram_score(self.synonyms, self.token_sent, self.pos_sent, self.index))
			synonym_scores['grammaticality'] = hf.get_gram_score(self.synonyms, self.token_sent, self.pos_sent, self.index)
			#filtering process
			synonym_scores = synonym_scores[synonym_scores['sem_sim']<0.15]
			synonym_scores = synonym_scores.sort_values(by=['complexity'])

			
			try:
				top_synomym = synonym_scores.synonyms.values[0]
				
			except:
				return [self.word]

			return top_synomym
		else:
			return [self.word]


	def get_synonym_complexities(self):
		if len(self.synonyms) > 1:
			synonym_scores = pd.DataFrame()
			synonym_scores['synonyms'] = self.synonyms
			synonym_scores['sem_sim'] = hf.get_elmo_score(self.synonyms, self.token_sent, self.index)
			synonym_scores['complexity'] = complex_word.get_synonym_complexities(self.synonyms, self.token_sent, self.index)
			synonym_scores['grammaticality'] = hf.get_gram_score(self.synonyms, self.token_sent, self.pos_sent, self.index)
			
			#filtering process, return top word?
			synonym_scores = synonym_scores[synonym_scores['sem_sim']<0.15]
			synonym_scores = synonym_scores.sort_values(by=['complexity'])

			return list(zip(synonym_scores['synonyms'].values,synonym_scores['complexity'].values))
		else:
			return None
		

	def get_synonym_dataframe(self):
		if len(self.synonyms) > 1:
			synonym_scores = pd.DataFrame()
			synonym_scores['synonyms'] = self.synonyms
			synonym_scores['sem_sim'] = hf.get_elmo_score(self.synonyms, self.token_sent, self.index)
			synonym_scores['complexity'] = complex_word.get_synonym_complexities(self.synonyms, self.token_sent, self.index)
			synonym_scores['grammaticality'] = hf.get_gram_score(self.synonyms, self.token_sent, self.pos_sent, self.index)
			
		
			synonym_scores = synonym_scores[synonym_scores['sem_sim']<0.3]
			#Can filter to only replace with words of lower threshold complexity
			#synonym_scores = synonym_scores[synonym_scores['complexity']<0.6]
			synonym_scores = synonym_scores[synonym_scores['grammaticality']==1]
			synonym_scores['combo'] = synonym_scores['sem_sim'] + synonym_scores['complexity']
			synonym_scores = synonym_scores.sort_values(by=['combo'])


			return synonym_scores
		else:
			return None
