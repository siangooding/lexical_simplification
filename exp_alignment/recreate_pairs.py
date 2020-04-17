# -*- coding: utf-8 -*-

if __name__ == '__main__':

    PATH_ADV_ELE_ADV = '../giza-pre-process/adv-ele-common/onestop.en-en.adv.orig'
    PATH_ADV_ELE_ELE = '../giza-pre-process/adv-ele-common/onestop.en-en.ele.orig'
    PATH_ADV_INT_ADV = '../giza-pre-process/adv-int-common/onestop.en-en.adv.orig'
    PATH_ADV_INT_INT = '../giza-pre-process/adv-int-common/onestop.en-en.int.orig'

    lines_adv_ele_adv = open(PATH_ADV_ELE_ADV).readlines()
    lines_adv_ele_ele = open(PATH_ADV_ELE_ELE).readlines()
    lines_adv_int_adv = open(PATH_ADV_INT_ADV).readlines()
    lines_adv_int_int = open(PATH_ADV_INT_INT).readlines()

    common_lines = set(lines_adv_ele_adv).intersection(set(lines_adv_int_adv))
    print(common_lines)

    f_adv_ele_adv = open('../giza-pre-process/adv-ele-common/onestop.en-en.adv', 'w')
    f_adv_ele_ele = open('../giza-pre-process/adv-ele-common/onestop.en-en.ele-common', 'w')
    f_adv_int_adv = open('../giza-pre-process/adv-int-common/onestop.en-en.adv', 'w')
    f_adv_int_int = open('.//giza-pre-process/adv-int-common/onestop.en-en.int-common', 'w')

    for i, line in enumerate(lines_adv_ele_adv):
        if line in common_lines:
            f_adv_ele_adv.write(line)
            f_adv_ele_ele.write(lines_adv_ele_ele[i])
    
    for i, line in enumerate(lines_adv_int_adv):
        if line in common_lines:
            f_adv_int_adv.write(line)
            f_adv_int_int.write(lines_adv_int_int[i])
    
    f_adv_ele_adv.close()
    f_adv_ele_ele.close()
    f_adv_int_adv.close()
    f_adv_int_int.close()

    lines_1 = open('../giza-pre-process/adv-ele-common/onestop.en-en.adv', 'r').readlines()
    lines_2 = open('../giza-pre-process/adv-ele-common/onestop.en-en.ele-common', 'r').readlines()
    lines_3 = open('../giza-pre-process/adv-int-common/onestop.en-en.adv', 'r').readlines()
    lines_4 = open('../giza-pre-process/adv-int-common/onestop.en-en.int-common', 'r').readlines()
    print(len(lines_1), len(lines_2), len(lines_3), len(lines_4))