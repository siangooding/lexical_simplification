from simplification import Sentence
from simplification import Word

def recursive_simplification(tokenized, threshold = 0.5, ignore_list = []):
	
	sentence_object = Sentence(tokenized,threshold,ignore_list)

	if (len(sentence_object.complex_words) > 0): #if there are complex words in the sentence
		
		#take the most complex word
		(index,complexity), *tail = sentence_object.complex_words
		
		#create word object 
		word_object = Word(sentence_object, index)
	
		#get the synonyms for this word
		word_object.get_synonyms()

		#rank these synonyms
		synonym = word_object.get_ranked_synonyms()

		if synonym != []:
			sentence_object.make_simplification(synonym, word_object.index)

		#recursively call function
		return recursive_simplification(sentence_object.tokenized,threshold, sentence_object.ignore_index)
	else:
		#when no simplifications possible return the sentence
		return sentence_object.tokenized


import nltk

sentence = nltk.word_tokenize("Sustainability is a big issue in the contemporary world ")
print(recursive_simplification(sentence))
#output: ['Sustainability', 'is', 'a', 'big', 'issue', 'in', 'the', 'modern', 'world']

sentence = nltk.word_tokenize("Motorists are being urged to take extra care in treacherous conditions.")
print(recursive_simplification(sentence))
#output: ['Motorists', 'are', 'being', 'advised', 'to', 'take', 'extra', 'care', 'in', 'dangerous', 'conditions', '.']
