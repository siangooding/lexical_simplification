# -*- coding: utf-8 -*-
import pickle
import argparse
from run_mapping_onestop import STORING
from settings.settings import ROOT_PATH

PICKLE_PATH = ROOT_PATH + STORING
LEVEL_TO_CEPP = {'1': 'KET', '2': 'PET', '3': 'FCE', '4': 'CAE', '5': 'CPE'}
CEPP_LEVELS = {'KET': '1', 'PET': '2', 'FCE': '3', 'CAE': '4', 'CPE': '5'}

class HistData:
    def __init__(self):
        self.t_path = pickle.load(open(ROOT_PATH + STORING.format('T'), 'rb'))
        self.l_path = pickle.load(open(ROOT_PATH + STORING.format('L'), 'rb'))
        self.sgr_path = pickle.load(open(ROOT_PATH + STORING.format('SGR'), 'rb'))
        self.ner_path = pickle.load(open(ROOT_PATH + STORING.format('NER'), 'rb'))
        self.all_path = pickle.load(open(ROOT_PATH + STORING.format('T_L_SGR_NER'), 'rb'))
        
        self.feat_to_path = {'t': self.t_path, 'l': self.l_path, 'sgr': self.sgr_path,
                             'ner': self.ner_path, 'all': self.all_path}
        self.feat_order = ['t', 'l', 'sgr', 'ner', 'all']
    
    def get_data(self, onestop_level):
        all_data = {feat: [0]*5 for feat in self.feat_order}
        for feat in self.feat_order:
            stored_info = self.feat_to_path[feat][onestop_level]
            for _, file_info in stored_info.items():
                level = int(file_info['level'])
                all_data[feat][level-1] += 1
        return all_data
    
    def autolabel(self, rects, ax):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    def plot_save_hist(self, onestop_level):
        bar_data = self.get_data(onestop_level)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.style.use('seaborn-deep')
        import numpy as np

        labels = ["KET", "PET", "FCE", "CAE", "CPE"]
        x = np.arange(len(labels))  # the label locations
        width = 0.2  # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(x - 0.4, bar_data["t"], width, label='ADV-INT')
        rects2 = ax.bar(x - 0.2, bar_data["l"], width, label='ADV-INT')
        rects3 = ax.bar(x , bar_data["sgr"], width, label='ADV-ELE')
        rects4 = ax.bar(x + 0.2, bar_data["ner"], width, label='INT-ELE')
        rects5 = ax.bar(x + 0.4, bar_data["all"], width, label='INT-ELE')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Number of texts')
        ax.set_title('Distribution of texts across levels and features')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        self.autolabel(rects1, ax)
        self.autolabel(rects2, ax)
        self.autolabel(rects3, ax)
        self.autolabel(rects4, ax)
        self.autolabel(rects5, ax)

        fig.tight_layout()
        plt.savefig(ROOT_PATH + 'figures/distrib_{0}.png'.format(onestop_level))


if __name__ == '__main__':
    # Assuming run from final_exp_mapping folder
    # python run_analyse_mapping.py -f T_L_SGR_NER -t table
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--features", required=True, help="features to consider, separated by `_`" +
                                                            "currently implemented: T_L_SGR_NER")
    ap.add_argument("-o", "--onestop_level", help="For `hist` type, OneStop level." +
                                                  "Should be `adv`, `int` or `ele`")
    ap.add_argument("-t", "--type", required=True, help="Type of analysis to do." +
                                                        "Either `table` for table numbers" +
                                                        "or `hist` for plotting histograms")
    args = vars(ap.parse_args())

    if args["type"] == 'table':
        mapping = pickle.load(open(PICKLE_PATH.format(args["features"]), 'rb'))
        for level, files_info in mapping.items():
            print("===== Results for {0} =====".format(level.upper()))
            tot = len(files_info)
            counts = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0}
            for file_path, curr_file_info in files_info.items():
                counts[str(curr_file_info['level'])] + 1
            for level in sorted(counts.keys()):
                print("{0} - {1}: {2}\t {3}".format(level, LEVEL_TO_CEPP[level], counts[level], round(100*counts[level]/tot, 2)))

    elif args["type"] == 'hist':
        hist_data = HistData()
        hist_data.plot_save_hist(args["onestop_level"])
    else:
        pass