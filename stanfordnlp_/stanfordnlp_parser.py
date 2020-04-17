# -*- coding: utf-8 -*-
import yaml
import argparse
import os
from os import listdir
import stanfordnlp
import xml.etree.ElementTree as ET

with open('./stanfordnlp_/config.yaml') as file:
    config_global = yaml.load(file, Loader=yaml.FullLoader)

config_stanford_nlp = config_global['config_stanford_nlp']

""" All the default languages are set to English and parameters + documentation accordingly.
Please refer to the parser's documentation for the use of another language. """


class StanfordNLP():
    """ 
    End-to-end pipeline for StanfordNLP Parser- English
    config='default'       => Default configuration : includes tokenisation
    config='pre_tokenized' => Option available : input sentence already tokenized 

    Processors summary
    Tokenize - POS - Lemma
    """
    def __init__(self, lang='en', config='pre_tokenized'):
        super().__init__()
        # Downloads the language models for the neural pipeline if never installed before
        if 'stanfordnlp_resources' not in listdir(config_stanford_nlp['general_dir']):
            stanfordnlp.download('en')
        # Initialize pipeline
        self.nlp = stanfordnlp.Pipeline(**config_stanford_nlp[config])
    
    def get_lines(self, lines_path):
        lines = []
        with open(lines_path, encoding='utf8') as fp:
            for _, line in enumerate(fp):
                lines.append(line[:-1])
        return lines
    
    def get_xml_path(self, lines_path, folder_xml):
        file_name_chunks = os.path.basename(lines_path).split('.')
        if file_name_chunks[-1] == 'txt':
            file_name_no_ext = '.'.join(os.path.basename(lines_path).split('.')[:-1])
        else:
            file_name_no_ext = '.'.join(os.path.basename(lines_path).split('.'))
        return folder_xml + '/' + file_name_no_ext + '.xml.stanfordnlp'
    
    def write_xml_format(self, lines_path, folder_xml):
        stanford_nlp_xml_path = self.get_xml_path(lines_path, folder_xml)
        lines = self.get_lines(lines_path)

        # Create XML file for final complete XML format
        root = ET.Element('article')
        root.text = '\n'
        root.tail = '\n\n'

        # Getting text content from initial lines
        celldata = ET.SubElement(root, 'text')
        celldata.text = '\n'    # newline before the collected element
        celldata.tail = '\n\n'  # empty line after the celldata element
        for line in lines:
            celldata.text += '{0} \n'.format(line)
        
        # Getting POS+Lemma (among others) with StanfordNLP
        celldata = ET.SubElement(root, 'stanfordnlp')
        celldata.text = '\n'
        celldata.tail = '\n'
        lines = [line.split(' ') for line in lines]
        for index, sent in enumerate(lines):
            curr_doc = self.nlp([sent])
            curr_sent = ET.SubElement(celldata, 'sentence', {'num': str(index+1)})
            curr_sent.text = '\n'
            curr_sent.tail = '\n'
            curr_word_list = ET.SubElement(curr_sent, 'word-list')
            curr_word_list.text = '\n'
            curr_word_list.tail = '\n'
            for word in curr_doc.sentences[0].words:
                attrib = {'index': str(word.index), 'text': str(word.text), 'lemma': str(word.lemma),
                          'upos': str(word.upos), 'xpos': str(word.xpos)}
                curr_word = ET.SubElement(curr_word_list, 'word', attrib)
                curr_word.tail = '\n'
        
        # Writing XML file
        tree = ET.ElementTree(root)
        tree.write(stanford_nlp_xml_path, encoding='utf-8', xml_declaration=True)
        return self


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="Input file - Raw text format ")
    ap.add_argument("-o", "--output", required=True, help="Output folder - StanfordNLP processed ")
    args = vars(ap.parse_args())

    PARSER = StanfordNLP()
    PARSER.write_xml_format(lines_path=args["input"],
                            folder_xml=args["output"])
    
    # Toy example, assuming .txt file in ./stanfordnlp folder
    # python ./stanfordnlp/stanfordnlp_parser.py -i ./stanfordnlp/test.txt -o ./stanfordnlp/ 

    # For aligned corpus and GIZA++
    # python ./stanfordnlp/stanfordnlp_parser.py -i ./giza-pre-process/onestop.en-en.tok.adv -o ./giza-pre-process  
    # python ./stanfordnlp/stanfordnlp_parser.py -i ./giza-pre-process/onestop.en-en.tok.ele -o ./giza-pre-process 
