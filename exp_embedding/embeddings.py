# -*- coding: utf-8 -*-
import argparse
import pickle
import numpy as np
import gensim
from tqdm import tqdm
from collections import defaultdict
from exp_alignment.explore_aligned import different_simplification
from exp_alignment.explore_aligned import get_levelled_words

# GLOBAL NUMBERS
PATH_ADV_INT = '../exp_alignment/levelled_words/adv-int.pkl'
PATH_ADV_ELE = '../exp_alignment/levelled_words/adv-ele.pkl'
PATH_INT_ELE = '../exp_alignment/levelled_words/int-ele.pkl'
adv_int = pickle.load(open(PATH_ADV_INT, 'rb'))
adv_ele = pickle.load(open(PATH_ADV_ELE, 'rb'))
int_ele = pickle.load(open(PATH_INT_ELE, 'rb'))

LEVELS = {"adv_int": adv_int, "adv_ele": adv_ele, "int_ele": int_ele}

class Embedding:
    def __init__(self, type_embedding):
        """ Type must be within ['word2vec', 'glove','fasttext'] """
        self.model_embedding = {'glove': 'glove-wiki-gigaword-200',
						        'word2vec': 'word2vec-google-news-300',
                                'fasttext': 'fasttext-wiki-news-subwords-300'}
        self.path_to_model = {'word2vec': './corpus_embedding/word2vec/word2vec.model',
                              'fasttext': './corpus_embedding/fasttext/fasttext.model'}

        if type_embedding == "glove":  
            import gensim.downloader as api
            self.model = api.load(self.model_embedding[type_embedding])
        
        if type_embedding == "word2vec":
            self.model = gensim.models.word2vec.Word2Vec.load(self.path_to_model["word2vec"])
        
        if type_embedding == "fasttext":
            self.model = gensim.models.fasttext.FastText.load(self.path_to_model["fasttext"])
    
    def get_embedding(self, lemma):
        try:
            return self.model.wv[lemma]
        except Exception:
            return None
    
    def get_most_similar(self, lemma, topn=10):
        try:
            return self.model.wv.most_similar(lemma, topn=topn)
        except Exception:
            return None
    
    def get_similarity(self, lemma1, lemma2):
        try:
            return self.model.wv.similarity(lemma1, lemma2)
        except Exception:
            return None


def update_embeddings(embedder, embeddings, word):
    word = word.split('_')[0]
    if word not in embeddings.keys():
        try:
            embeddings[word] = embedder.get_embedding(word)
        except Exception:  # Most probable = OOV
            embeddings[word] = None
    return embeddings


def validate_word(embeddings, word):
    if embeddings[word] is None:
        return False
    return len(embeddings[word]) > 0


def clean_levelled_words(levelled_words, embeddings):
    res = []
    for elt in levelled_words:
        if validate_word(embeddings, elt[0].split('_')[0]) and validate_word(embeddings, elt[1]) and validate_word(embeddings, elt[2]):
            if elt[1] != elt[2]:  # different simplification for INT and ELE levels
                res.append(elt)
    return res


def get_sim(embedder, correspondences):
    sim = []
    oov = 0
    for higher_word in correspondences.keys():
        for lower_word in set(correspondences[higher_word]):
            sim_attempt = embedder.get_similarity(higher_word.split('_')[0], lower_word)
            if sim_attempt is not None:
                sim.append(sim_attempt)
            else:
                oov += 1
    return sim, oov


def get_top(level, embedder, embeddings):
    print("Analysing most similar words for level: {0}".format(level))
    data = LEVELS[level]
    data_cleaned = {word.split('_')[0]:simp for word, simp in data.items() if embeddings[word.split('_')[0]] is not None}
    print("Number of original found words: {0}".format(len(data)))
    print("Number of words with embedding: {0}".format(len(data_cleaned)))

    count = 0
    top_10, top_20, top_50, top_100 = 0, 0, 0, 0
    for higher_word, simp in data_cleaned.items():
        most_similar = embedder.get_most_similar(lemma=higher_word, topn=100)
        if most_similar is not None:
            most_similar = [elt[0] for elt in most_similar]
            for lower_word in set(simp):
                count += 1
                if lower_word in most_similar[:10]:
                    top_10 += 1
                    top_20 += 1
                    top_50 += 1
                    top_100 += 1
                if lower_word in most_similar[10:20]:
                    top_20 += 1
                    top_50 += 1
                    top_100 += 1
                if lower_word in most_similar[20:50]:
                    top_50 += 1
                    top_100 += 1
                if lower_word in most_similar[50:100]:
                    top_100 += 1
    print("Number of lower level words: {0}".format(count))
    print("Number of words which were in the complex's word top 10: {0}".format(top_10))
    print("Number of words which were in the complex's word top 20: {0}".format(top_20))
    print("Number of words which were in the complex's word top 50: {0}".format(top_50))
    print("Number of words which were in the complex's word top 100: {0}".format(top_100))
    print("===============")

    return [top_10, top_20, top_50, top_100, count]


if __name__ == '__main__':
    # All python commands are assumed to be run from ./exp_embedding/ folder
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--type", required=True, help="Type of operations to do " + 
                                                        "`save`: saving word embeddings for all levels" +
                                                        "`distrib` : distribution of similarity ranges" +
                                                        "`gap` : gap in similarities between different simplifications " +
                                                        "for the same word" +
                                                        "`most_similar` : exploring distribution of simplifications : are they " +
                                                        "in top10 most similar, top20, etc " +
                                                        "`2d_vis` : Visualisations in 2D"
                                                        "`load`")
    ap.add_argument("-e", "--embedding", required=True, help="Embedding to use" + 
                                                             "`word2vec`, `fasttext`, `glove`")
    ap.add_argument("-pca", "--pca", default="0", help="Number of components for PCA" +
                                                         "if 0 no PCA ")  
    ap.add_argument("-d", "--data", default="all", help="Type of data to consider " + 
                                                        "`all` : all simplifications for all levels " ) 
    ap.add_argument("-l", "--level", default=None, help="Type of level to consider for exploring gap in similarities" +
                                                        "`adv_int`, `adv_ele`, `int_ele`")   
    ap.add_argument("-n", "--normalized", default="0", help="Normalize histogram or not. Boolean")
    ap.add_argument("-w", "--words", default=None, help="Pre-selected words to display in 2D. Those words must have an embeddings" + 
                                                        "and be in the aligned words found by GIZA++ " + 
                                                        "yaml file containing keys : complex, int, ele")
    ap.add_argument("-wf", "--word_filter", default=None, help="For 2D vis, displaying only one POS or not" +
                                                             "Most popular encountered : `NOUN`, `VERB`, `ADJ`")          
    args = vars(ap.parse_args())

    if args["type"] == 'save':
        # python embeddings.py -t save -e word2vec -pca 0
        # python embeddings.py -t save -e word2vec -pca 2
        # python embeddings.py -t save -e fasttext -pca 0
        # python embeddings.py -t save -e fasttext -pca 2
        # python embeddings.py -t save -e glove -pca 0 
        # python embeddings.py -t save -e glove -pca 2 
        pkl_path = './stored_embeddings/embeddings_{0}_{1}.pkl'.format(args["embedding"], args["pca"])
        embedder = Embedding(args["embedding"])
        embeddings = defaultdict(list)

        words = set(adv_int).union(set(adv_ele)).union(set(int_ele))
        for word in tqdm(words):
            embeddings = update_embeddings(embedder, embeddings, word)

        if args["pca"] != '0':
            from sklearn import decomposition
            import numpy as np
            words = [word for word in embeddings.keys() if embeddings[word] is not None]
            print('Original words: {0}, Have an embedding: {1}'.format(len(embeddings.keys()), len(words)))
            X = np.asarray([embeddings[word] for word in words if embeddings[word] is not None])
            pca = decomposition.PCA(n_components=int(args["pca"]))
            X_PC2 = pca.fit(X).transform(X)
            for index, word in enumerate(words):
                embeddings[word] = X_PC2[index]

        pickle.dump(embeddings, open(pkl_path, 'wb'))
    

    if args["type"] == "distrib":
        # python embeddings.py -t distrib -e glove -pca 0 -d all 
        # python embeddings.py -t distrib -e fasttext -pca 0 -d all 
        # python embeddings.py -t distrib -e word2vec -pca 0 -d all 
        # Ensuring we consider original embeddings and not projected ones
        pkl_path = './stored_embeddings/embeddings_{0}_{1}.pkl'.format(args["embedding"], "0")
        embedder = Embedding(args["embedding"])
        embeddings = pickle.load(open(pkl_path, 'rb'))

        # Getting data for histogram
        sim_adv_int, oov = get_sim(embedder, correspondences=adv_int)
        print("OOV for ADV-INT (at least one word) : {0}".format(oov))
        sim_adv_ele, oov = get_sim(embedder, correspondences=adv_ele)
        print("OOV for ADV-ELE (at least one word) : {0}".format(oov))
        sim_int_ele, oov = get_sim(embedder, correspondences=int_ele)
        print("OOV for INT-ELE (at least one word) : {0}".format(oov))

        # Plotting histogram
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.style.use('seaborn-deep')
        nb_bins = 10
        bins = np.linspace(0, 1, nb_bins+1)
        plt.hist([sim_adv_int, sim_adv_ele, sim_int_ele],
                    bins, label=['ADV-INT', 'ADV-ELE', 'INT-ELE'], normed=int(args["normalized"]))
        plt.legend()
        plt.title('Similarity distributions for {0} data'.format(args["data"]))
        plt.savefig('./figures/similarity_distrib_{0}_{1}_{2}.png'.format(args["embedding"],args["data"], args["normalized"]))

        # Gettings hist counts
        import numpy as np
        data = [sim_adv_int, sim_adv_ele, sim_int_ele]
        legends = ["ADV-INT", "ADV-ELE", "INT-ELE"]
        for i, d in enumerate(data):
            counts, _ = np.histogram(d, bins=bins)
            print(legends[i])
            print("{0}\n ==========".format(counts))
    

    if args["type"] == "gap":
        # python embeddings.py -t gap -e glove -pca 0 -l adv_int
        # python embeddings.py -t gap -e glove -pca 0 -l adv_ele
        # python embeddings.py -t gap -e glove -pca 0 -l int_ele
        # python embeddings.py -t gap -e word2vec -pca 0 -l adv_int
        # python embeddings.py -t gap -e word2vec -pca 0 -l adv_ele
        # python embeddings.py -t gap -e word2vec -pca 0 -l int_ele
        # python embeddings.py -t gap -e fasttext -pca 0 -l adv_int
        # python embeddings.py -t gap -e fasttext -pca 0 -l adv_ele
        # python embeddings.py -t gap -e fasttext -pca 0 -l int_ele
    
        # Ensuring we consider original embeddings and not projected ones
        pkl_path = './stored_embeddings/embeddings_{0}_{1}.pkl'.format(args["embedding"], "0")
        embedder = Embedding(args["embedding"])
        embeddings = pickle.load(open(pkl_path, 'rb'))

        import numpy as np 
        simp_per_word, data = [], LEVELS[args["level"]]
        for word, simp in data.items():
            simp_per_word.append(len(set(simp)))
        more_than_one = [elt for elt in simp_per_word if elt > 1]
        print("Analysis for level : {0}".format(args["level"]))
        print("Number of higher level words : {0}".format(len(data.keys()))) 
        print("Number of lower level words : {0}".format(np.sum(np.array(simp_per_word))))
        print("Number of higher level words with stricly more than one simplifications : {0}".format(len(more_than_one))) 
        print("Average nb of simplification / word : {0}".format(np.mean(np.array(simp_per_word))))
        print("Average nb of simplification / word (only 2 or more simp) : {0}".format(np.mean(np.array(more_than_one))))

        words_with_more_than_one_simp = {word:simp for word, simp in data.items() if len(set(simp)) > 1}
        gap_similarities = []
        for word, simp in words_with_more_than_one_simp.items():
            sim, oov = get_sim(embedder, correspondences={word:simp})
            gap_similarities.append({'sim': sim, 'oov': oov})
        avg_gap_similarities_per_word = np.array([np.mean(np.array(elt["sim"])) for elt in gap_similarities])
        #print("Average gap similarities per word \n {0}".format(avg_gap_similarities_per_word))
        print("Overall gap similarities average: {0}".format(np.mean(avg_gap_similarities_per_word)))
        print("Overall gap similarities std: {0}".format(np.std(avg_gap_similarities_per_word)))


    if args["type"] == "most_similar":
        # python embeddings.py -t most_similar -e fasttext -pca 0
        # python embeddings.py -t most_similar -e glove -pca 0
        # python embeddings.py -t most_similar -e word2vec -pca 0 
        # Ensuring we consider original embeddings and not projected ones
        pkl_path = './stored_embeddings/embeddings_{0}_{1}.pkl'.format(args["embedding"], "0")
        embedder = Embedding(args["embedding"])
        embeddings = pickle.load(open(pkl_path, 'rb'))

        bar_data = {"adv_int": None, "adv_ele": None, "int_ele": None}
        for level in ["adv_int", "adv_ele", "int_ele"]:
            data = get_top(level, embedder, embeddings)
            bar_data[level] = data
        
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.style.use('seaborn-deep')

        labels = ["top10", "top20", "top50", "top100", "total"]
        x = np.arange(len(labels))  # the label locations
        width = 0.3  # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(x - 0.3, bar_data["adv_int"], width, label='ADV-INT')
        rects2 = ax.bar(x , bar_data["adv_ele"], width, label='ADV-ELE')
        rects3 = ax.bar(x + 0.3, bar_data["int_ele"], width, label='INT-ELE')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Number of words')
        ax.set_title('Number of words in the top n most similar words of the complex word ')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()


        def autolabel(rects):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')


        autolabel(rects1)
        autolabel(rects2)
        autolabel(rects3)

        fig.tight_layout()
        plt.savefig('./figures/most_similar_{0}.png'.format(args["embedding"]))
    

    if args["type"] == "2d_vis":
        # python embeddings.py -t 2d_vis -e glove -pca 2 
        pkl_path = './stored_embeddings/embeddings_{0}_{1}.pkl'.format(args["embedding"], "2")
        embeddings = pickle.load(open(pkl_path, 'rb'))

        if args["words"] is None:
            levelled_words = get_levelled_words(different=different_simplification,
                                                adv_to_int=adv_int, adv_to_ele=adv_ele)
            levelled_words = clean_levelled_words(levelled_words, embeddings)
            # print(levelled_words)
            nb_triplets = len(levelled_words)

            if args["word_filter"] is None:
                complex_words = [elt[0].split('_')[0] for elt in levelled_words[:nb_triplets]]
                int_words = [elt[1].split('_')[0] for elt in levelled_words[:nb_triplets]]
                ele_words = [elt[2].split('_')[0] for elt in levelled_words[:nb_triplets]]
            else:
                # python embeddings.py -t 2d_vis -e glove -pca 2 -wf NOUN 
                pos = [elt[0].split('_')[1] for elt in levelled_words[:nb_triplets]]
                complex_words, int_words, ele_words = [], [], []
                for i in range(len(pos)):
                    if pos[i] == args["word_filter"]:
                        complex_words.append(levelled_words[i][0].split('_')[0])
                        int_words.append(levelled_words[i][1].split('_')[0])
                        ele_words.append(levelled_words[i][2].split('_')[0])

        
        else:
            # python embeddings.py -t 2d_vis -e glove -pca 2 -w words.yaml 
            import yaml
            with open(args["words"]) as file:
                params = yaml.load(file, Loader=yaml.FullLoader)
            complex_words, int_words, ele_words = params["complex"], params["int"], params["ele"]
        
        nb_triplets = len(complex_words)
        selected_words = complex_words + int_words + ele_words

        x, y = [], []
        for word in selected_words:
            #print(embeddings[word])
            x.append(embeddings[word][0])
            y.append(embeddings[word][1])
        
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.style.use('seaborn-deep')
        colors = ['red'] * nb_triplets + ['blue'] * nb_triplets + ['green'] * nb_triplets
        plt.scatter(x, y, c=colors)

        for i, word in enumerate(selected_words):
            plt.annotate(word, (x[i], y[i]))
        
        for i in range(nb_triplets):
            complex_word, int_word, ele_word = complex_words[i], int_words[i], ele_words[i]
            [x_complex, y_complex] = list(embeddings[complex_word]) 
            [x_int, y_int] = list(embeddings[int_word])
            [x_ele, y_ele] = list(embeddings[ele_word])
            plt.plot([x_complex, x_int], [y_complex, y_int], 'b')
            plt.plot([x_int, x_ele], [y_int, y_ele], 'g')

        plt.savefig('./figures/2d_vis_{0}_{1}.png'.format(args["embedding"],args["data"]))
