# -*- coding: utf-8 -*-
import pickle
import numpy as np
import nltk
from nltk.corpus import stopwords
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
with open('./distrib.pickle', 'rb') as f:
    distrib = pickle.load(f)

stop_words = set(stopwords.words('english'))
print(distrib.keys())

levels = distrib.keys()
distrib_mean = {}
for level in levels:
    distrib_mean[level] = {'sent': {}, 'text': {}}
    for type in ['sent', 'text']:
        for word in distrib[level][type].keys():
            if word not in stop_words:  # removing English stopwords
                distrib_mean[level][type][word] = [np.mean(np.array(distrib[level][type][word]))]

def get_list_hist(distrib, distrib_mean, level):
    hist_sent_ind = []
    hist_sent_mean = []
    hist_text_ind = []
    hist_text_mean = []

    for _, complexities in distrib[level]['sent'].items():
        hist_sent_ind += complexities
    for _, complexities in distrib[level]['text'].items():
        hist_text_ind += complexities
    for _, complexities in distrib_mean[level]['sent'].items():
        hist_sent_mean += complexities
    for _, complexities in distrib_mean[level]['sent'].items():
        hist_text_mean += complexities

    return np.array(hist_sent_ind), np.array(hist_sent_mean), np.array(hist_text_ind), np.array(hist_text_mean)

def get_hist_data_pos(distrib, distrib_mean, level, pos):
    hist_sent_ind = []
    hist_sent_mean = []
    hist_text_ind = []
    hist_text_mean = []

    for word, complexities in distrib[level]['sent'].items():
        if nltk.pos_tag([word])[0][1] in pos:
            hist_sent_ind += complexities
    for word, complexities in distrib[level]['text'].items():
        if nltk.pos_tag([word])[0][1] in pos:
            hist_text_ind += complexities
    for word, complexities in distrib_mean[level]['sent'].items():
        if nltk.pos_tag([word])[0][1] in pos:
            hist_sent_mean += complexities
    for word, complexities in distrib_mean[level]['sent'].items():
        if nltk.pos_tag([word])[0][1] in pos:
            hist_text_mean += complexities

    return np.array(hist_sent_ind), np.array(hist_sent_mean), np.array(hist_text_ind), np.array(hist_text_mean)

hist = {}
for LEVEL in distrib.keys():
    hist_sent_ind, hist_sent_mean, hist_text_ind, hist_text_mean = get_list_hist(distrib, distrib_mean, LEVEL)
    hist[LEVEL] = {}
    hist[LEVEL]['hist_sent_ind'] = hist_sent_ind
    hist[LEVEL]['hist_sent_mean'] = hist_sent_mean
    hist[LEVEL]['hist_text_ind'] = hist_text_ind
    hist[LEVEL]['hist_text_mean'] = hist_text_mean

HIST_NOUN = {}
POS_NOUN = ['NN', 'NNS', 'NNP', 'NNPS']
HITS_VB = {}
POS_VB = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
HIST_ADJ = {}
POS_ADJ = ['JJ', 'JJR', 'JJS']

for LEVEL in distrib.keys():
    hist_sent_ind, hist_sent_mean, hist_text_ind, hist_text_mean = get_hist_data_pos(distrib, distrib_mean, LEVEL, POS_NOUN)
    HIST_NOUN[LEVEL] = {}
    HIST_NOUN[LEVEL]['hist_sent_ind'] = hist_sent_ind
    HIST_NOUN[LEVEL]['hist_sent_mean'] = hist_sent_mean
    HIST_NOUN[LEVEL]['hist_text_ind'] = hist_text_ind
    HIST_NOUN[LEVEL]['hist_text_mean'] = hist_text_mean

    hist_sent_ind, hist_sent_mean, hist_text_ind, hist_text_mean = get_hist_data_pos(distrib, distrib_mean, LEVEL, POS_VB)
    HITS_VB[LEVEL] = {}
    HITS_VB[LEVEL]['hist_sent_ind'] = hist_sent_ind
    HITS_VB[LEVEL]['hist_sent_mean'] = hist_sent_mean
    HITS_VB[LEVEL]['hist_text_ind'] = hist_text_ind
    HITS_VB[LEVEL]['hist_text_mean'] = hist_text_mean

    hist_sent_ind, hist_sent_mean, hist_text_ind, hist_text_mean = get_hist_data_pos(distrib, distrib_mean, LEVEL, POS_ADJ)
    HIST_ADJ[LEVEL] = {}
    HIST_ADJ[LEVEL]['hist_sent_ind'] = hist_sent_ind
    HIST_ADJ[LEVEL]['hist_sent_mean'] = hist_sent_mean
    HIST_ADJ[LEVEL]['hist_text_ind'] = hist_text_ind
    HIST_ADJ[LEVEL]['hist_text_mean'] = hist_text_mean

# for LEVEL in distrib.keys():
#     hist_sent_ind, hist_sent_mean, hist_text_ind, hist_text_mean = get_list_hist(distrib, distrib_mean, LEVEL)
#     fig,a =  plt.subplots(2,2)
#     fig.tight_layout()
#     plt.style.use('seaborn-deep')
#     a[0][0].hist(hist[LEVEL]['hist_sent_ind'])
#     a[0][0].set_title('{0} - Sent level - ind'.format(LEVEL))
#     a[0][1].hist(hist[LEVEL]['hist_sent_mean'])
#     a[0][1].set_title('{0} - Sent level - mean'.format(LEVEL))
#     a[1][0].hist(hist[LEVEL]['hist_text_ind'])
#     a[1][0].set_title('{0} - Text level - ind'.format(LEVEL))
#     a[1][1].hist(hist[LEVEL]['hist_text_mean'])
#     a[1][1].set_title('{0} - Text level - mean'.format(LEVEL))
#     plt.savefig('./figures_distrib/fig_1_{0}.png'.format(LEVEL))
#     #plt.show()

# plot_to_hist = {
#     0: {0: {'hist': 'hist_sent_ind', 'title': 'Sent level - Ind'},
#         1: {'hist': 'hist_sent_mean', 'title': 'Sent level - Mean'}},
#     1: {0: {'hist': 'hist_text_ind', 'title': 'Text level - Ind'},
#         1: {'hist': 'hist_text_mean', 'title': 'Text level - Mean'}}
# }

# plt.style.use('seaborn-deep')

# fig,a =  plt.subplots(2,2, figsize=(10,10))
# fig.tight_layout()
# bins = np.linspace(0, 1, 10)

# for x_pos in plot_to_hist.keys():
#     for y_pos in plot_to_hist.keys():
#         ket = hist['KET'][plot_to_hist[x_pos][y_pos]['hist']]
#         pet =hist['PET'][plot_to_hist[x_pos][y_pos]['hist']]
#         fce = hist['FCE'][plot_to_hist[x_pos][y_pos]['hist']]
#         cae = hist['CAE'][plot_to_hist[x_pos][y_pos]['hist']]
#         cpe = hist['CPE'][plot_to_hist[x_pos][y_pos]['hist']]

#         a[x_pos][y_pos].hist([ket, pet, fce, cae, cpe], bins, label=['ket', 'pet', 'fce', 'cae', 'cpe'
#         ])
#         a[x_pos][y_pos].legend(loc='upper center')
#         a[x_pos][y_pos].set_title(plot_to_hist[x_pos][y_pos]['title'])
# plt.savefig('./figures_distrib/fig_2.png')
# #plt.show()


# plot_to_hist = {
#     0: {0: {'hist': 'hist_sent_ind', 'title': 'Sent level - Ind'},
#         1: {'hist': 'hist_sent_mean', 'title': 'Sent level - Mean'}},
#     1: {0: {'hist': 'hist_text_ind', 'title': 'Text level - Ind'},
#         1: {'hist': 'hist_text_mean', 'title': 'Text level - Mean'}},
# }
# plt.style.use('seaborn-deep')

# fig,a =  plt.subplots(2,2, figsize=(10,10))
# fig.tight_layout()
# bins = np.linspace(0, 1, 20)

# for x_pos in plot_to_hist.keys():
#     for y_pos in plot_to_hist.keys():
#         ket = hist['KET'][plot_to_hist[x_pos][y_pos]['hist']]
#         pet =hist['PET'][plot_to_hist[x_pos][y_pos]['hist']]
#         fce = hist['FCE'][plot_to_hist[x_pos][y_pos]['hist']]
#         cae = hist['CAE'][plot_to_hist[x_pos][y_pos]['hist']]
#         cpe = hist['CPE'][plot_to_hist[x_pos][y_pos]['hist']]

#         a[x_pos][y_pos].hist([ket, pet, fce, cae, cpe], bins, label=['ket', 'pet', 'fce', 'cae', 'cpe'
#         ])
#         a[x_pos][y_pos].legend(loc='upper center')
#         a[x_pos][y_pos].set_title(plot_to_hist[x_pos][y_pos]['title'])
# plt.savefig('./figures_distrib/fig_3.png')
# #plt.show()


def get_std_complexities(distrib_level, level):
    std_val, std_nb = [], 0
    for _, c in distrib_level.items():
        if len(c) > 1:
            std_val.append(np.std(np.array(c)))
            std_nb += 1
    print('STD Mean : {0}\t STD non null nb : {1}\t Level : {2}'.format(np.mean(np.array(std_val)), std_nb, level))

levels = ['KET', 'PET', 'FCE', 'CAE', 'CPE']
print('TEXT LEVEL')
for level in levels:
    get_std_complexities(distrib_level=distrib[level]['text'], level=level)
print('SENT LEVEL')
for level in levels:
    get_std_complexities(distrib_level=distrib[level]['sent'], level=level)


y_to_hist = {0: HIST_NOUN, 1: HIST_ADJ, 2: HITS_VB}

plot_to_hist = {
    0: {0: {'hist': 'hist_text_mean', 'title': 'NOUN'},
        1: {'hist': 'hist_text_mean', 'title': 'ADJ'},
        2: {'hist': 'hist_text_mean', 'title': 'VERB'}},
    # 1: {0: {'hist': 'hist_text_ind', 'title': 'Text level - Ind - Noun'},
    #     1: {'hist': 'hist_text_mean', 'title': 'Text level - Mean - Noun'}},
}
plt.style.use('seaborn-deep')

fig,a =  plt.subplots(1,3, figsize=(10,5))
fig.tight_layout()
bins = np.linspace(0, 1, 10)

for x_pos in plot_to_hist.keys():
    for y_pos in plot_to_hist[x_pos].keys():
        ket = y_to_hist[y_pos]['KET'][plot_to_hist[x_pos][y_pos]['hist']]
        pet =y_to_hist[y_pos]['PET'][plot_to_hist[x_pos][y_pos]['hist']]
        fce = y_to_hist[y_pos]['FCE'][plot_to_hist[x_pos][y_pos]['hist']]
        cae = y_to_hist[y_pos]['CAE'][plot_to_hist[x_pos][y_pos]['hist']]
        cpe = y_to_hist[y_pos]['CPE'][plot_to_hist[x_pos][y_pos]['hist']]

        a[y_pos].hist([ket, pet, fce, cae, cpe], bins, label=['ket', 'pet', 'fce', 'cae', 'cpe'
        ])
        a[y_pos].legend(loc='upper center')
        a[y_pos].set_title(plot_to_hist[x_pos][y_pos]['title'])
plt.savefig('./figures_distrib/fig_4.png')
#plt.show()


