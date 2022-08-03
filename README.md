# Lexical Simplification 

This is a system developed for lexical simplification based on the paper by [Gooding and Kochmar (2019)](https://www.aclweb.org/anthology/D19-1491.pdf). Given a sentence the system will identify complex words and recursively simplify the lexical content in a context-aware manner. 

## Requirements

Requirements can be found in requirements.txt and installed using:

```python
pip install -r requirements.txt
```

You will need to generate your own API key for Big Huge Theasurus: https://words.bighugelabs.com/site/api

This key goes in simplification.py where THEASURUS_KEY is defined. 

## Usage
The recursive simplification function is defined in [recursive_simplification.py](./recursive_simplification.py) alongside examples of applying the system to sentences. 

## System Output

The system output on the test sets from [Gooding and Kochmar (2019)](https://www.aclweb.org/anthology/D19-1491.pdf) for the original system is available in [sytem output](./system%20output).


## Citing

If you use our model, please kindly cite our paper:

```
@inproceedings{gooding-kochmar-2019-recursive,
    title = "Recursive Context-Aware Lexical Simplification",
    author = "Gooding, Sian  and
      Kochmar, Ekaterina",
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
