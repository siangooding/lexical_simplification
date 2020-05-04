# -*- coding: utf-8 -*-
from collections import defaultdict
import pickle

PATH_ADV_INT = './levelled_words/adv-int.pkl'
PATH_ADV_ELE = './levelled_words/adv-ele.pkl'
PATH_INT_ELE = './levelled_words/int-ele.pkl'

def get_info_upos(words, message):
    upos = defaultdict(int)
    for complex_word in words:
        upos[complex_word.split('_')[-1]] += 1
    print(message)
    for upos_val, nb in upos.items():
        print('{0}\t{1}\t{2}'.format(upos_val, nb, round(100 * nb/len(words), 2)))

# GLOBAL NUMBERS
adv_int = pickle.load(open(PATH_ADV_INT, 'rb'))
adv_ele = pickle.load(open(PATH_ADV_ELE, 'rb'))
int_ele = pickle.load(open(PATH_INT_ELE, 'rb'))

# COMPARING ADV_INT AND ADV_ELE
simp_for_int = set(adv_int.keys())
simp_for_ele = set(adv_ele.keys())
simp_for_int_ele = set(int_ele.keys())
nb_adv_int, nb_adv_ele, nb_int_ele = 0, 0, 0
for key in simp_for_int:
    nb_adv_int += len(set(adv_int[key]))
for key in simp_for_ele:
    nb_adv_ele += len(set(adv_ele[key]))
for key in simp_for_int_ele:
    nb_int_ele += len(set(int_ele[key]))
commons = simp_for_int.intersection(simp_for_ele)

common_simplification = set()
different_simplification = set()
for complex_word in commons:
    # if len(set(adv_int[complex_word]).intersection(set(adv_ele[complex_word]))) > 0:
    #     common_simplification.add(complex_word)
    # else:
    #     different_simplification.add(complex_word)
    if set(adv_int[complex_word]) == set(adv_ele[complex_word]):
        common_simplification.add(complex_word)
    else:
        different_simplification.add(complex_word)


def get_levelled_words(different, adv_to_int, adv_to_ele):
    """ Takes all advanced words whose simplifications were different in INT
    and ELE level and returns a list of list 
    [[a, b, c]] with a the word in the ADV text, b the aligned word in the
    INT level and c the aligned word in the ELE level """
    res = []
    for adv_word in sorted(list(different)):
        for int_word in adv_to_int[adv_word]:
            for ele_word in adv_to_ele[adv_word]:
                if [adv_word, int_word, ele_word] not in res:
                    res.append([adv_word, int_word, ele_word])
    return res



if __name__ == '__main__':
    print('# of simplifications from ADV to INT :\t {0} unique, \t {1} distinct'.format(len(adv_int), nb_adv_int))
    print('# of simplifications from ADV to ELE :\t {0} unique, \t {1} distinct'.format(len(adv_ele), nb_adv_ele))
    print('# of simplifications from INT to ELE :\t {0} unique, \t {1} distinct'.format(len(int_ele), nb_int_ele))
    print('====================')


    print('# of common advanced words simplified :\t {0}'.format(len(commons)))
    print("{0}% of INT simplifications, {1}% of ELE simplifications".format(round(100*len(commons)/len(simp_for_int), 2), round(100*len(commons)/len(simp_for_ele), 2)))


    get_info_upos(words=common_simplification,
              message="'Repartition of POS - advanced words with at least one common simplification'")
    print('# of advanced words with different simplifications :\t {0}'.format(len(different_simplification)))
    print('{0}% of common advanced words simplified'.format(round(100*len(different_simplification)/len(commons), 2)))
    get_info_upos(words=different_simplification,
                message="'Repartition of POS - advanced words with different simplification'")
    print('====================')

    print("Levelled simplification")
    for complex_word in different_simplification:
        print('{0}\t {1}\t {2}'.format(complex_word, list(set(adv_int[complex_word])), list(set(adv_ele[complex_word]))))
    print('====================')

    # SIMPLIFIED FOR INT LEVEL ONLY (CONTRARY TO INTUITION)
    # possible explaination : content simply removed in the elementary version
    print('# of advanced words simplified for INT only :\t {0}'.format(len(adv_int) - len(commons)))
    print('{0}% of complex words simplified to int'.format(round(100 *(len(adv_int) - len(commons))/len(adv_int), 2)))
    #print(simp_for_int.difference(commons))
    get_info_upos(words=simp_for_int.difference(commons),
                message='Repartition of POS - simplified for INT only')
    print('====================')

    # SIMPLIFIED FOR ELE LEVEL ONLY (SHOULD BE INT WORDS, OR NOT PRESENT/MATCHED IN INT LEVEL)
    print('# of advanced words simplified for ELE only :\t {0}'.format(len(adv_ele) - len(commons)))
    print('{0}% of complex words simplified to ele'.format(round(100 * (len(adv_ele) - len(commons))/len(adv_ele), 2)))
    get_info_upos(words=simp_for_ele.difference(commons),
                message='Repartition of POS - simplified for ELE only')
    print('====================')

    # LEVELLED WORDS
    levelled_words = get_levelled_words(different=different_simplification,
                                        adv_to_int=adv_int, adv_to_ele=adv_ele)
    print(len(levelled_words))