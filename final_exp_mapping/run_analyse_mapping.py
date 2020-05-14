# -*- coding: utf-8 -*-
import pickle
import argparse
from settings.settings import ROOT_PATH

STORING = 'final_exp_mapping/mappings/mapping_one_stop_{0}.pkl'
PICKLE_PATH = ROOT_PATH + STORING
LEVEL_TO_CEPP = {'1': 'KET', '2': 'PET', '3': 'FCE', '4': 'CAE', '5': 'CPE'}
CEPP_LEVELS = {'KET': '1', 'PET': '2', 'FCE': '3', 'CAE': '4', 'CPE': '5'}

class HistData:
    def __init__(self):
        self.t = pickle.load(open(ROOT_PATH + STORING.format('T'), 'rb'))
        self.l = pickle.load(open(ROOT_PATH + STORING.format('L'), 'rb'))
        self.sgr = pickle.load(open(ROOT_PATH + STORING.format('SGR'), 'rb'))
        self.ner = pickle.load(open(ROOT_PATH + STORING.format('NER'), 'rb'))
        self.all = pickle.load(open(ROOT_PATH + STORING.format('T_L_SGR_NER'), 'rb'))
        
        self.feat_to_path = {'t': self.t, 'l': self.l, 'sgr': self.sgr,
                             'ner': self.ner, 'all': self.all}
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
        width = 0.15  # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(x - 0.3, bar_data["t"], width, label='T')
        rects2 = ax.bar(x - 0.15, bar_data["l"], width, label='L')
        rects3 = ax.bar(x , bar_data["sgr"], width, label='SGR')
        rects4 = ax.bar(x + 0.15, bar_data["ner"], width, label='NER')
        rects5 = ax.bar(x + 0.3, bar_data["all"], width, label='ALL')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Number of texts')
        ax.set_title('Distribution of {0} OneStop texts across CEFR levels'.format(onestop_level.upper()))
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        self.autolabel(rects1, ax)
        self.autolabel(rects2, ax)
        self.autolabel(rects3, ax)
        self.autolabel(rects4, ax)
        self.autolabel(rects5, ax)

        fig.tight_layout()
        plt.savefig(ROOT_PATH + 'final_exp_mapping/figures/distrib_{0}.png'.format(onestop_level))


if __name__ == '__main__':
    # Assuming run from final_exp_mapping folder
    # python run_analyse_mapping.py -f T_L_SGR_NER -t table
    # python run_analyse_mapping.py -o adv -t hist
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--features", help="features to consider, separated by `_`" +
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
                counts[str(curr_file_info['level'])] += 1
            for level in sorted(counts.keys()):
                print("{0} - {1}: {2}\t {3}".format(level, LEVEL_TO_CEPP[level], counts[level], round(100*counts[level]/tot, 2)))

    elif args["type"] == 'hist':
        hist_data = HistData()
        hist_data.plot_save_hist(args["onestop_level"])
    else:
        pass