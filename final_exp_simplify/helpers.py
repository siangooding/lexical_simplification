# -*- coding: utf-8 -*-py

def load_data(feature_file):
    lines = open(feature_file).readlines()
    data, targets = [], []
    for a_line in lines:
        targets.append(float(a_line.split()[0]))
        attributes = a_line.split()[1:]
        input_vector = [float(attribute.split(':')[1]) for attribute in attributes]
        data.append(input_vector)
    return data, targets