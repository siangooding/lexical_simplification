# -*- coding: utf-8 -*-py
import numpy as np
import nltk

def tokenize_text(file_path, save_path):
    f_save = open(save_path, "w+")
    with open(file_path) as f:
        for line in f:
            tokenized_sentence = nltk.word_tokenize(line)
            f_save.write('{0}\n'.format(' '.join(tokenized_sentence)))
    f_save.close()


def get_numbers_orig_text(file_path):
    lines = open(file_path).readlines()
    lines = [line.replace('\n', '').split(' ') for line in lines]
    lines = [len(line) for line in lines]
    print('Total numbers of lines: {0}'.format(len(lines)))
    print('Average number of tokens per sentence: {0}'.format(np.mean(np.array(lines))))


def compare_len_sentences(file_path_1, file_path_2, file_path_3):
    for file_path in [file_path_1, file_path_2, file_path_3]:
        lines = open(file_path).readlines()
        lines = [line.replace(' \n', '').split(' ') for line in lines]
        lines = [len(line) for line in lines]
        print("Number of tokens for: {0}\n{1}".format(file_path, lines))
    print("=====")


def get_simplified_tokens(original_file, simplified_file):
    print("Comparing: {0} and {1}".format(original_file.split('/')[-1].split('.')[0],
                                          simplified_file.split('/')[-1].split('.')[0]))
    lines_orig = [line.replace(' \n', '').split(' ') for line in open(original_file).readlines()]
    lines_simp = [line.replace(' \n', '').split(' ') for line in open(simplified_file).readlines()]

    simp_tokens = 0
    tot_tokens = 0
    for index_line, line_orig in enumerate(lines_orig):
        for index_token, token in enumerate(line_orig):
            if token != lines_simp[index_line][index_token]:
                simp_tokens += 1
            tot_tokens += 1
    
    print("Total number of tokens: {0}".format(tot_tokens))
    print("Number of tokens simplified: {0}".format(simp_tokens))
    print("==========")


def highlight_simplifications(original_file, simplified_file, most_simplified_file, f_save=NotImplementedError):
    if f_save is not None:
        f_save.write("Comparing: {0} and {1}\n\n".format(original_file.split('/')[-1].split('.')[0],
                                          simplified_file.split('/')[-1].split('.')[0]))
    print("Comparing: {0} and {1}\n".format(original_file.split('/')[-1].split('.')[0],
                                          simplified_file.split('/')[-1].split('.')[0]))
    lines_orig = [line.replace('\n', '').split(' ') for line in open(original_file).readlines()]
    lines_simp = [line.replace('\n', '').split(' ') for line in open(simplified_file).readlines()]
    lines_most = [line.replace('\n', '').split(' ') for line in open(most_simplified_file).readlines()]

    for index_line, line_orig in enumerate(lines_orig):
        for index_token, token in enumerate(line_orig):
            simplified = False
            index_begin = max(index_token - 4, 0)
            index_end = min(index_token + 4, len(line_orig))

            if token != lines_simp[index_line][index_token]:
                simplified = True
                if f_save is not None:
                    f_save.write("Original:\t\t {0}\n".format(' '.join(line_orig[index_begin:index_end])))
                    f_save.write("0.8 threshold:\t {0}\n".format(' '.join(lines_simp[index_line][index_begin:index_end])))
                print("Original:\t {0}".format(' '.join(line_orig[index_begin:index_end])))
                print("0.8 threshold:\t {0}".format(' '.join(lines_simp[index_line][index_begin:index_end])))

            if token != lines_most[index_line][index_token]:
                if not simplified:
                    if f_save is not None:
                        f_save.write("Original:\t\t {0}\n".format(' '.join(line_orig[index_begin:index_end])))
                    print("Original:\t {0}".format(' '.join(line_orig[index_begin:index_end])))
                simplified = True
                if f_save is not None:
                    f_save.write("0.5 threshold:\t {0}\n".format(' '.join(lines_most[index_line][index_begin:index_end])))
                print("0.5 threshold:\t {0}".format(' '.join(lines_most[index_line][index_begin:index_end])))
            
            if simplified:
                if f_save is not None:
                    f_save.write("=====\n")
                print("=====")
    
    if f_save is not None:
        f_save.write("\n##################################################\n")
    print("\n##################################################\n")



if __name__ == '__main__':
    file_path = './onestop/adv-text/{0}-adv.txt'
    save_path = './onestop/simplified/{0}-adv-{1}.txt'
    thresholds = ['10.0', '8.0', '5.0']

    f_save = open("./onestop/simplified/simplifications.txt", "w")
    for name in ['Amazon', 'Amsterdam', 'Anita', 'Arctic-mapping', 'Banksy']:
        pass
        # tokenize_text(file_path=file_path.format(name), save_path=save_path.format(name, '10.0'))
        # get_numbers_orig_text(file_path=save_path.format(name, "10.0"))
        # compare_len_sentences(file_path_1=save_path.format(name, '10.0'),
        #                       file_path_2=save_path.format(name, '8.0'),
        #                       file_path_3=save_path.format(name, '5.0'))
        # get_simplified_tokens(original_file=save_path.format(name, '10.0'),
        #                       simplified_file=save_path.format(name, '8.0'))
        # get_simplified_tokens(original_file=save_path.format(name, '10.0'),
        #                       simplified_file=save_path.format(name, '5.0'))

        highlight_simplifications(original_file=save_path.format(name, "10.0"),
                                  simplified_file=save_path.format(name, "8.0"),
                                  most_simplified_file=save_path.format(name, "5.0"),
                                  f_save=f_save)
    f_save.close()

