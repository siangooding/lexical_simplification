# Recursive Lexical Simplification 

This software is a lexical simplification system, modified from its original version that can be found on Sian's repository, [lexical_simplification](https://github.com/siangooding/lexical_simplification) It was built upon the system presented by [Gooding and Kochmar (2019)](https://www.aclweb.org/anthology/D19-1491.pdf). 

Given a text the system will optionnally simplify this text at a given level, depending on the input parameters

## Usage

The main file containing the recursive simplification system is [recursive_simplification](./recursive_simplification.py). One example of configuration file is given in the [config file](./recursive_config.yaml). The different parameters are described in the [recursive_simplification](./recursive_simplification.py). Overall, several types of systems can be implemented:

* _Baseline_. The original system presented by [Gooding and Kochmar (2019)](https://www.aclweb.org/anthology/D19-1491.pdf).
* _Pre selection_. A pre selection score was added as the selection step after the generation step: the top 5 highly ranked synonyms are selected for the ranking step.
* _Cascade_. A cascade score is added the _Pre selection_ sysyem described above. The most salient words are first selected as words to be simplified.
* _Combined_. A combined score is added the _Pre selection_ sysyem described above. The words to be simplified are first selected according to a combination of salience and complexity.

## Structure

* [ngram](./ngram)
* [system output](./system_output)
* [verb](./verb)
* [cached_synonyms](./cached_synonyms.yaml)
* [complex_labeller](./complex_labeller.py)
* [complex_word](./complex_word.py)
* [conlleval](./conlleval.py)
* [evaluator](./evaluator.py)
* [experiment](./experiment.py)
* [gpu_attention.model](./gpu_attention.model)
* [helper_functions](./helper_functions.py)
* [labeler](./labeler.py)
* [plural](./plural.py)
* [recursive_config](./recursive_config.yaml)
* [recursive_simplification](./recursive_simplification.py)
* [scores](./scores.py): new incorporated scores for the different steps within the lexical simplification. ImportanceSimplifyScore and SelectionScore.
* [simplification](./simplification): Word and Sentence classes
* [thesaurus](./thesaurus.py)
* [traditional_features](./traditional_features.py): saving frequency of words within CEFR corpora.

## Citing

The following paper was used as a starting point for building the new lexical simplification system:

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
```
