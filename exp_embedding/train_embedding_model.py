# -*- coding: utf-8 -*-
import argparse 
import gensim
import smart_open
from datetime import datetime
from os import listdir
from gensim import utils

# News Crawl 2011 : https://www.statmt.org/wmt11/
# EW & SEW : https://cs.pomona.edu/~dkauchak/simplification/
# SubIMDB : https://ghpaetzold.github.io/subimdb/
# UMBC : https://ebiquity.umbc.edu/resource/html/id/351

LINE_TO_CONTENT = {
    "ew_sew": lambda line: line.split('\t')[2].replace('\n', ''),
    "news_crawl": lambda line: line,
    "subimdb": lambda line: line,
    "umbc": lambda line: line
}


def read_compiled_corpus(corpora_path='./corpus_embedding/'):
    corpora = [elt for elt in listdir(corpora_path) if 'gz' not in elt and "word2vec" not in elt and "fasttext" not in elt]
    for corpus in corpora:
        files = [elt for elt in listdir(corpora_path+corpus) if 'README' not in elt and 'possf2' not in elt]
        for file_name in files:
            with smart_open.open(corpora_path+corpus+'/'+file_name, encoding="iso-8859-1") as f:
                for _, line in enumerate(f):
                    yield utils.simple_preprocess(LINE_TO_CONTENT[corpus](line))


def train_save(model, save_model, corpora_path='./corpus_embedding/'):
    date_begin = datetime.now()
    print("Began process at {0}".format(date_begin))
    train_corpus = list(read_compiled_corpus(corpora_path=corpora_path))
    print("Built corpus, finished at {0}".format(datetime.now()))
    model.build_vocab(train_corpus)
    print("Built vocabulary, finished at {0}".format(datetime.now()))
    print("Size vocabulary: {0}".format(model.corpus_count))
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
    print("Trained model, finished at {0}".format(datetime.now()))
    model.save(save_model)
    print("Saved model, finished at {0}".format(datetime.now()))
    print("Full process took {0}".format(datetime.now() - date_begin))


if __name__ == '__main__':
    # To be run from ./exp_embedding/ folder 
    # python train_embedding_model.py -m word2vec -s ./corpus_embedding/word2vec/word2vec.model
    # python train_embedding_model.py -m fasttext -s ./corpus_embedding/fasttext/fasttext.model
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model_type", required=True, help="type of model to train " + 
                                                              "`word2vec` or `fasttext`")
    ap.add_argument("-s", "--save", required=True, help="saving path for embedding model ")
    args = vars(ap.parse_args())

    if args["model_type"] not in ["word2vec", "fasttext"]:
        print("Wrong type of model entered, try again")

    else:
        if args["model_type"] == "word2vec":
            from gensim.models.word2vec import Word2Vec
            model = Word2Vec(sg=0, size=650)
        
        if args["model_type"] == "fasttext":
            from gensim.models.fasttext import FastText
            model = FastText(sg=0, size=300)
        
        train_save(model=model, save_model=args["save"])
