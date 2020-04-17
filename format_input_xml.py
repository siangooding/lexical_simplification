# -*- coding: utf-8 -*-py
import os
import subprocess
import argparse
import xml.etree.ElementTree as ET

def convert_txt_to_simple_xml(text_path, folder_xml):
    with open(text_path) as f:
        celldata = ET.Element('text')
        celldata.text = '\n'    # newline before the collected element
        celldata.tail = '\n\n'  # empty line after the celldata element
        for line in f:
            celldata.text += line 

    # Include the root element to the tree and write the tree to the file.
    tree = ET.ElementTree(celldata)
    file_name_chunks = os.path.basename(text_path).split('.')
    if file_name_chunks[-1] == 'txt':
        output_file = folder_xml + '.'.join(os.path.basename(text_path).split('.')[:-1]) + '.xml'
    else:
        output_file = folder_xml + '.'.join(os.path.basename(text_path).split('.')) + '.xml'
    tree.write(output_file, encoding='utf-8', xml_declaration=True)


def format_text_to_input(simple_xml_path, folder_xml,
                         rasp_path='./readability_code/rasp3os/scripts/rasp.sh'):
    file_name = os.path.basename(simple_xml_path)
    shell_cmd = "{0} ".format(rasp_path) + \
                "-x '-p text' -wp '-oGTC " + \
                "-l {0}.log' < {1} > {0}.rasptemp 2> {0}.err".format(folder_xml+file_name, simple_xml_path)
    subprocess.call(shell_cmd, shell=True)
    if os.path.exists('{0}.log'.format(folder_xml+file_name)):
        os.remove('{0}.log'.format(folder_xml+file_name))
    if os.path.exists('{0}.err'.format(folder_xml+file_name)):
        os.remove('{0}.err'.format(folder_xml+file_name))


def create_complete_xml(text_path, folder_xml):

    file_name_chunks = os.path.basename(text_path).split('.')
    if file_name_chunks[-1] == 'txt':
        file_name_no_ext = '.'.join(os.path.basename(text_path).split('.')[:-1])
    else:
        file_name_no_ext = '.'.join(os.path.basename(text_path).split('.'))
    
    simple_xml_path = folder_xml + file_name_no_ext + '.xml'
    rasp_xml_path = folder_xml + file_name_no_ext + '.xml.rasptemp'

    convert_txt_to_simple_xml(text_path=text_path, 
                              folder_xml=folder_xml) # Convert .txt to .xml with text only
    format_text_to_input(simple_xml_path=folder_xml+file_name_no_ext+'.xml', 
                         folder_xml=folder_xml)  # Create .xml.rasptemp with rasp

    # Create XML file for the final complete XML file
    root= ET.Element('xml')

    root_article = ET.SubElement(root, 'article')
    root_article.text = '\n'

    ## Getting text content from the simple xml path
    celldata = ET.SubElement(root_article, 'text')
    for _, element in ET.iterparse(simple_xml_path):
        if element.tag == "text":
            celldata.text = element.text
            celldata.tail = '\n\n'
    
    ## Getting rasp content from the complete xml path
    celldata = ET.SubElement(root_article, 'rasp')
    et_rasp = ET.parse(rasp_xml_path).getroot()
    celldata.extend(et_rasp)

    # Writing and deleting unused files
    tree = ET.ElementTree(root)
    tree.write(folder_xml+file_name_no_ext+'.xml.rhaps', encoding='utf-8', xml_declaration=True)
    os.remove('{0}.xml'.format(folder_xml+file_name_no_ext))
    os.remove('{0}.xml.rasptemp'.format(folder_xml+file_name_no_ext))


if __name__ == '__main__':
    # This script will take all .txt files in a given folder and creates the right input for the pipeline ##
    # 1 = creating a .xml file with text only
    # 2 = creating a .xml.rasptemp file with rasp output only
    # 3 = combining the two xml trees and removing all previously created files
    # Example for command line (assuming in root directory)
    # python format_input_xml.py -i ./data/original/3d-indoormap.en.0.txt -o ./data/formatted/
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="Input file - Raw text format ")
    ap.add_argument("-o", "--output", required=True, help="Output file - RASP parsed ")
    args = vars(ap.parse_args())

    create_complete_xml(text_path=args['input'], folder_xml=args['output'])
    
    # For aligned corpus and GIZA ++
    # python format_input_xml.py -i ./exp_alignment/giza-pre-process/onestop.en-en.adv -o ./exp_alignment/giza-pre-process/
    # python format_input_xml.py -i ./exp_alignment/giza-pre-process/onestop.en-en.ele -o ./exp_alignment/giza-pre-process/

