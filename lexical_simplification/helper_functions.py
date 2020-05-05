import yaml
import numpy as np
from scipy.spatial.distance import cosine
from settings.settings import ROOT_PATH

def get_elmo_score(synonyms, tok_sentence, index, elmo):
	
	vectors = elmo.embed_sentence(tok_sentence)
	original_vector = vectors[2][index]

	distances = []

	for synonym in synonyms:
		new_sentence = tok_sentence.copy()
		del new_sentence[index]
		for i,word in enumerate(synonym):
			new_sentence.insert((index + i), word)

		new_vectors = elmo.embed_sentence(new_sentence)
		if len(synonym) == 1:
			new_vector = new_vectors[2][index]
		else:
			phrase_vectors = []
			for i,word in enumerate(synonym):
				phrase_vectors.append(np.array(new_vectors[2][(index + i)]))
			new_vector = np.mean(phrase_vectors,axis=0)

		
		distances.append(cosine(original_vector, new_vector))

	return distances


def get_ngram(left, right):
	import pandas as pd
	two_grams = pd.read_csv(ROOT_PATH + 'lexical_simplification/ngram/w2_.txt', sep='\t', header=None, encoding='latin1')
	two_grams.columns = ["freq", "w1", "w2"]

	freq = two_grams[(two_grams["w1"]==left) & (two_grams["w2"]==right)].freq.values

	return freq


def get_gram_score(synonyms, tokenized, pos_tags, index):


	left_bi = True
	right_bi = True

	original_word = tokenized[index]

	try:
		left_word = tokenized[index - 1]
		left_pos = pos_tags[index - 1]
	except:
		left_bi = False

	try:
		right_word = tokenized[index + 1]
		right_pos = pos_tags[index + 1]
	except:
		right_bi = False

	if right_word in [',','.',';',':'] or ('P' in right_pos[1]):
		right_bi = False

	if left_word in [',','.',';',':'] or ('P' in left_pos[1]):
		left_bi = False

	original_left = get_ngram(left_word,original_word)
	

	if  original_left.size == 0:
		left_bi = False
	if get_ngram(original_word,right_word).size == 0:
		right_bi = False



	for synonym in synonyms:
		if len(synonym) == 1:
			if left_bi:
				left_prob = get_ngram(left_word, synonym[0])
			if right_bi:
				right_prob = get_ngram(synonym[0], right_word)
		else:
			synonym_left = synonym[len(synonym)-1]
			synonym_right = synonym[0]

			if left_bi:
				left_prob = get_ngram(left_word, synonym_right)
			if right_bi:
				right_prob = get_ngram(synonym_left, right_word)

		if left_bi and right_bi:
			
			if (np.sum(left_prob) > 0) and (np.sum(right_prob) > 0):
			    return 1
			else:
				return 0
		elif not left_bi:
			if right_bi:
				if right_prob > 0:
					return 1
				else:
					return 0

		elif not right_bi:
			if left_bi:
				if left_prob > 0:
					return 1
				else:
					return 0


def different_than_one_edit(s1, s2):
	nb_edit = 0
	for i, char_1 in enumerate(s1):
		if char_1.lower() != s2[i].lower():
			nb_edit += 1
	return nb_edit != 1


def more_than_one_insert(longer, shorter):
	index_l, index_s = 0, 0
	n_l, n_s = len(longer), len(shorter)
	nb_difference, found  = 0, False
	while (nb_difference <= 1) and (not found) and (index_l < n_l) and (index_s < n_s):
		if longer[index_l].lower() == shorter[index_s].lower():
			index_l += 1
			index_s += 1
		else:
			nb_difference += 1
			index_l += 1
			found = True
	
	if nb_difference > 2:
		return True
	return not longer[index_l+1:].lower() == shorter[index_s+1:].lower()


def is_not_too_similar(candidate, original):
	""" Checking if a word and a substitution candidate are not too alike
	Two words are considered too alike if there is only one of the following operations
	between the two words : insert, edit, delete """
	if abs(len(candidate) - len(original)) > 1:
		return True
	
	elif len(candidate) == len(original):
		return different_than_one_edit(candidate, original)
	
	elif len(candidate) > len(original):
		return more_than_one_insert(candidate, original)
	
	else:
		return more_than_one_insert(original, candidate)


if __name__ == '__main__':
	s1 = 'violence'
	s2 = 'fight'
	print(is_not_too_similar(candidate=s2, original=s1))