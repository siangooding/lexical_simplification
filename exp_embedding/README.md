# Quantitative and qualitative analysis related to aligned words

-----------------------

I wanted to explore how words were simplified from one higher level text to a lower level text and if I could find any relation between aligned words. All experiments run with Python 3.7.4.

Resources :

* Datasets to train word embedding model :
  * Benchmarking Lexical Simplification Systems - Paetzold and Specia - 2016
  * News Crawl 2011 : <https://www.statmt.org/wmt11/>
  * EW & SEW : <https://cs.pomona.edu/~dkauchak/simplification/>
  * SubIMDB : <https://ghpaetzold.github.io/subimdb/>
  * UMBC : <https://ebiquity.umbc.edu/resource/html/id/351> (not used for training, simple pointer)

* Gensim Python library : <https://radimrehurek.com/gensim/>

## Word embeddings for aligned words

I compared and analysed different word embeddings for aligned words :

* Pre-trained glove-wiki-gigaword-200

* Re-trained Word2Vec model with 650 dimensions. Trained on News Crawl 2011 + EW/SEW + SubIMDB

* Re-trained fastText model with 300 dimensions. Trained on News Crawl 2011 + EW/SEW + SubIMDB

### Usage

* **Training a word embedding model.** The main file to reproduce my training is [train_embedding_model](train_embedding_model.py). Before using this script, you have to download the corresponding data on the different websites given as pointer. On the root folder I had a `corpus_embedding` folder with the following structure :

* [corpus_embedding](corpus_embedding)  : training corpus + models
  * [ew_sew](corpus_embedding/ew_sew) : English Wikipedia and Simple English Wikipedia
  * [fasttext](corpus_embedding/fasttext) : fastText model (currently just the model is here, if you want to retrain you can add the data in the same folder)
  * [news_crawl](corpus_embedding/news_crawl) : News Crawl 2011
  * [subimdb](corpus_embedding/subimdb) : SubIMDB
  * [word2vec](corpus_embedding/word2vec) : word2vec model (same than for fasttext)

Once you have trained your models, you can use the main file for analysing word embeddings, `embeddings`(embeddings.py). All commands given as example in the file are assumed run from root folder in terminal.

There are several parameters that can be set :

* `type` : type of operations to perform. Will be described just below
* `embedding` : embedding model to use. Either `glove`, `fasttext` or `word2vec`
* `pca` : number of dimensions if using PCA
* `data` : default `all`, for now only option available
* `level` : level to perform operation on. Either `adv_int`, `adv_ele` or `int_ele`
* `normalized` : to normalize similarity distribution histogram or not
* `words` : default `None`, else path to a `.yaml` file containing keys `complex`, `int` and `ele`, with a list of words as values. Lists must be of same list. One example is provided in [words](./words.yaml)
* `word_filter` : when using 2D-visualisations of words, filtering on POS or not

Depending on the type of command you use, you might need to create a `figures` folder using `mkdir figures` in this directory.

After are briefly described the different operations that can currently be done.

* **Saving word embeddings**. Saving word embeddings for all levels. Possibility to store vector after performing PCA, in this case provide the number of dimensions. Currently used for experiments : PCA dim = 0 (no PCA), PCA dim = 2 (for 2D visualisations)

* **Distribution of similarity ranges.** For each level difference (ADV-INT, ADV-ELE and INT-ELE) and given one embedding model, plotting histograms of similarity ranges between a complex word and its aligned simplified word.

* **Exploring gap in similarities.** Exploring gap in similarities between different simplifications for the same complex word.

* **Most similar words to complex words.** For each level difference and given one embedding model, plotting how many simplified words were in top 10, 20, 50 and 100 most similar word to their associated complex word.

* **2D visualisations.** Projecting word embeddings in 2D space to observe pattern. Red dots = ADV words, blue dots = INT words, green dots = ELE words. Blue line = mapping ADV word to its aligned INT word, green line = mapping INT word to its ELE word. Words considered : complex words which had a different aligned INT word and ELE word.
