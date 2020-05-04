# Lexical Simplification System

-----------------------

This software is a lexical simplification system. It was built upon the system presented by [Gooding and Kochmar (2019)](https://www.aclweb.org/anthology/D19-1491.pdf). Furthermore the readability assessment pipeline was built upon the readability classified introduced by [Xia et al (2016)](https://www.aclweb.org/anthology/W16-0502.pdf).

Given a text the system will optionnally simplify this text at a given level, and predict its ranking level using the readability classifier.

## Installation

All code was executed using Python 3.7.4 + Tensorflow 1.13.1, an NVIDIA GPU and a virtual environment.

First clone the project and install the `textrank` submodule properly:

```sh
git clone https://github.com/Blinines/lexical_simplification.git
```

Requirements can be found in the [requirements](./requirements.txt) and must be installed using:

```sh
pip install -r requirements.txt
```

First create a `private.py` file in the settings folder, and add the following elements (defining variables):

* `ROOT_PATH`: the absolute path to the folder containing this README.md.
* `API_KEY_BIG_HUGE_LABS`: your API key to retrieve the synonyms from Moby Thesaurus. For this, you will need to create an account and make an API key demand. For more details please visit the [Big Huge Thesaurus website](https://words.bighugelabs.com/).
* `RASP_PATH`: the absolute path to the RASP parser (`rasp.sh` file) within your machine. You can download the parser on the [iLexIR website](https://www.ilexir.co.uk/rasp/index.html).

Details of the whole package can be found in the [setup](./setup.py) and must be installed using:

```python
python setup.py install
```

Please note that if you reuse this code and modify some code within a module, you might have to rerun the latter command for the system to work.

The `textrank` folder was taken as a .zip file from its original [implementation](https://github.com/summanlp/textrank), branch d9252a233c93ec43693e0f145a025ae534b275b1. You need to install in particular the `summa` module:

```sh
cd textrank
python setup.py install
```

## Usage

The main file summing up the whole pipeline is [pipeline_readability](./pipeline_readability.py). Other folders or files either include helpers, modules or specific experiment. If looking more in detail at one experiment, `exp` should be in the folder name and a corresponding README in the folder.

## Structure

* [exp_alignment](./exp_alignment): using GIZA++, aligning sentences from OneStopEnglish parallel data and using the Stanford NLP Parser to extract specific candidates. See corresponding [README](./exp_alignment/README.md) for more details.
* [exp_distrib](./exp_distrib): exploring the distribution of word complexities using [Gooding and Kochmar (2019)](https://www.aclweb.org/anthology/D19-1491.pdf) system across levelled corpora. See corresponding [README](./exp_distrib/README.md) for more details.
* [exp_embedding](./exp_embedding): quantitative and qualitative analysis related to aligned words. Exploring embeddings between paired words at different levels. See corresponding [README](./exp_embedding/README.md) for more details.
* [exp_threshold](./exp_threshold): impact of threshold in the original lexical simplification system introduced in [Gooding and Kochmar (2019)](https://www.aclweb.org/anthology/D19-1491.pdf). See corresponding [README](./exp_threshold/README.md) for more details.
* [final_exp_mapping](./final_exp_mapping): to add if run in the end
* [final_exp_simplify](./final_exp_simplify): to add, final experiment for main task
* [format_input](./format_input): formatting a raw text file into an _.xml_ file, with the corresponding RASP output.
* [lexical_simplification](./lexical_simplification): For the original lexical simplification introduced in [Gooding and Kochmar (2019)](https://www.aclweb.org/anthology/D19-1491.pdf), please see [Sian's code](https://github.com/siangooding/lexical_simplification). I have built my new system upon this code.
* [onestop](./onestop): Some [OneStopEnglish texts]((https://www.aclweb.org/anthology/W18-0535.pdf)) used within my experiments. Includes aligned texts, simplified texts with different systems, and samples. All original texts were originally taken on the official [OneStopEnglish repository](https://github.com/nishkalavallabhi/OneStopEnglishCorpus).
* [readability_code](./readability_code): to add
* [settings](./settings)
* [stanfordnlp_](./stanfordnlp_): Related to the installation and usage of the StanfordNLP parser. See corresponding [README](./stanfordnlp_/README.md) for more details.
* [textrank](./textrank): For the original TextRank implementation I used, please see the corresponding [implementation](https://github.com/summanlp/textrank). I slightly modified some parts for the purposes of my experiments.

## Citing

The two following papers were used as a starting point for my dissertatioon:

```sh

@inproceedings{gooding-kochmar-2019-recursive,
    title = "Recursive Context-Aware Lexical Simplification",
    author = "Gooding, Sian  and Kochmar, Ekaterina",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D19-1491",
    doi = "10.18653/v1/D19-1491",
    pages = "4855--4865",
}

@article{xia2019text,
  title={"Text readability assessment for second language learners"},
  author={"Xia, Menglin and Kochmar, Ekaterina and Briscoe, Ted"},
  journal={arXiv preprint arXiv:1906.07580},
  year={2016}
}
```
