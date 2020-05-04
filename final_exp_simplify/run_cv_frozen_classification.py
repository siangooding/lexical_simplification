# -*- coding: utf-8 -*-py
from os import listdir
import numpy as np
from copy import deepcopy
from os.path import splitext
from datetime import datetime
from helpers import load_data
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

STORED_FEAT_PATH = '/home/ib431/Documents/projects/cam_mphil_project/readability_code/OneStop_features/'
ROOT_LOG_PATH = '/home/ib431/Documents/projects/cam_mphil_project/final_exp_simplify/logs_grid_search/'
POSSIBLE_FEATURES = ['T', 'L', 'SGR', 'NER']

class GridSearchLSParams:

    def __init__(self, type_model, features, stored_feat_path=STORED_FEAT_PATH,
                 root_log_path=ROOT_LOG_PATH):
        """ This cross-validation determines the best parameters for the lexical simplification system,
        while using the same parameters for the ML Pipeline
        All code assume that the features were already stored as .txt files
        All features stored are assumed to have `train` in their filename, in order to select them
        The CV retrieves all systems corresponding to the `type_model` and performs cross-validation (default 5)
        Current pipeline used for the classifider : MinMaxScaler + RandomForestClassifier
        @param type_model: to be chosen among `baseline`, `no_salience` (just pre selection of synonyms),
        `cascade` (salience with cascade + pre selection), `combined` (salience with combined + pre selection)
        @param features: features considered for the classifier. Current implemented features among 
        `T`, `L`, `SGR`, `NER`, must be separated by `_`
        @param: stored_feat_path: path to the stored features, as .txt files
        @param: root_log_path: path to store the logs and main info for the CV """
        self.type_model = type_model
        self.type_to_pattern = {'baseline': 'baseline', 'no_salience': 'no_salience',
                                'cascade': 'salience_cascade', 'combined': 'salience_combined'}
        self.features = features
        self.stored_feat_path = stored_feat_path
        self.all_systems = self.find_all_systems()
        self.load_all_data()

        self.f_log = '{0}{1}_{2}'.format(root_log_path, type_model, features)
    
    def validate_features(self, filename):
        splitted = splitext(filename)[0].split('.')[0].split('_')
        curr_features = [elt for elt in splitted if elt in POSSIBLE_FEATURES]
        if set(curr_features) == set(self.features.split('_')):
            return True
        return False

    def find_all_systems(self):
        filenames = listdir(self.stored_feat_path)
        pattern = self.type_to_pattern[self.type_model]
        selected = [filename for filename in filenames if pattern in filename and 'train' in filename]
        return [elt for elt in selected if self.validate_features(elt)]
    
    def build_cv_folders(self, y, nb=5):
        folders = [[] for i in range(5)]
        for level in [1, 2, 3]:
            indexes = [i for i, elt in enumerate(y) if elt == level]
            for folder_nb in range(nb):
                folders[folder_nb] += [elt for i, elt in enumerate(indexes) if i % nb == folder_nb]
        return folders
    
    def load_all_data(self):
        self.X_train_all, self.y_train_all = [], []
        for filename in self.all_systems:
            curr_X_train, curr_y_train = load_data(feature_file=self.stored_feat_path+filename)
            self.X_train_all.append(curr_X_train)
            self.y_train_all.append(curr_y_train)
    
    def sep_train_test(self, index_test, folders, X, y):
        new_X_train, new_y_train = None, None
        for i, elt in enumerate(folders):
            if i != index_test:
                new_X_train = X[elt] if new_X_train is None else np.concatenate((new_X_train, X[elt]))
                new_y_train = y[elt] if new_y_train is None else np.concatenate((new_y_train, y[elt]))

        return new_X_train, new_y_train, X[folders[index_test]], y[folders[index_test]]
    
    def write_log(self, type_log, info={}):
        save_log = open(self.f_log, 'a')
        if type_log == 'begin_all':
            save_log.write('CV began at {0} \n'.format(info['date']))
            save_log.write('Type model: {0}\t features: {1}\n'.format(self.type_model, self.features))
            save_log.write('===============')
        
        if type_log == 'begin_system':
            save_log.write('Number: {0}\t System: {1} \n'.format(info['index'], info['name']))
            save_log.write('Began at: {0} \n'.format(info['date']))
        
        if type_log == 'begin_index_test':
            save_log.write('\n Folder {0} as test folder, process began at {1} \n'
                .format(info["index_test"], info["date"]))
        
        if type_log == 'end_index_test':
            save_log.write('Finished fitting and predicting at {0}, took {1} \n'
                .format(info['date_end'], info['tot']))
            save_log.write('Confusion matrix: {0}\n'.format(info['confusion']))
            save_log.write('Accuracy: {0}\n'.format(info['acc']))
        
        if type_log == 'end_system':
            save_log.write('\n Finished processing system {0} at {1}, took {2}\n'
                .format(info['index'], info['date_end'], info['tot']))
            save_log.write('Best accuracy: {0}\n =============== \n'.format(info['best_acc']))
        
        if type_log == 'end_all':
            save_log.write('\n Finished processing all systems at {0}, took {1}\n'
                .format(info['date_end'], info['tot']))
            save_log.write('Best accuracy: {0} \n'.format(info['best_acc']))
            save_log.write('Best params: {0}\n'.format(info['best_params']))

        save_log.close()
    
    def cross_validate(self):
        date_begin_all = datetime.now()
        self.write_log(type_log='begin_all', info={'date': date_begin_all})
        best_acc_all, best_params = None, None

        for index, filename in enumerate(self.all_systems):  # Taking all possible systems (e.g. different LS params)
            date_begin_system = datetime.now()
            self.write_log(type_log='begin_system', info={'name': filename, 'index': index, 'date': date_begin_system})
            
            curr_X, curr_y = np.array(self.X_train_all[index]), np.array(self.y_train_all[index])
            cv_folders = self.build_cv_folders(curr_y)
            best_acc = None

            for index_test in range(len(cv_folders)):  # CV, with each folder taken as test folder
                date_begin_index_test = datetime.now()
                self.write_log(type_log='begin_index_test', info={'index_test': index_test, 'date': date_begin_index_test})
                
                curr_X_train, curr_y_train, curr_X_test, curr_y_test = \
                    self.sep_train_test(index_test, folders=cv_folders, X=curr_X, y=curr_y)
                
                text_clf = Pipeline([('min_max_scaler', preprocessing.MinMaxScaler()),
                                     ('clf', RandomForestClassifier(n_estimators=10, max_depth=None,
                                                                    min_samples_split=2, random_state=0)),]) 
                text_clf.fit(curr_X_train, curr_y_train)
                y_predicted = text_clf.predict(curr_X_test)
                
                confusion = confusion_matrix(curr_y_test, y_predicted)
                accuracy = np.mean(y_predicted==curr_y_test)

                date_end_index_test = datetime.now()
                self.write_log(type_log='end_index_test',
                               info={'date_end': date_end_index_test, 'tot': date_end_index_test - date_begin_index_test,
                                     'acc': accuracy, 'confusion': confusion})
                # update best accuracy for the given system
                best_acc = accuracy if best_acc is None else max(best_acc, accuracy)

            date_end_system = datetime.now()
            self.write_log(type_log='end_system', info={'date_end': date_end_system,
                                                        'tot': date_end_system - date_begin_system,
                                                        'best_acc': best_acc, 'index': index})
            
            # update overall best accuracy
            best_acc_all = best_acc if best_acc_all is None else max(best_acc_all, best_acc)
            best_params = filename if best_params is None else filename
            best_params = filename if best_acc > best_acc_all else best_params
        
        date_end_all = datetime.now()
        self.write_log(type_log='end_all', info={'date_end': date_end_all, 'tot': date_end_all - date_begin_all,
                                                 'best_params': best_params, 'best_acc': best_acc_all})


if __name__ == '__main__':
    grid_search = GridSearchLSParams(None, None)
    print(grid_search.build_cv_folders(y=[1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,3,3]))