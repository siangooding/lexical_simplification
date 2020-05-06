# Mapping OneStop texts to CEFR levels

-----------------------

The aim of this experiment was to map the OneStopEnglish corpus texts to the CEFR levels. The ranking system was first trained on CEFR levelled corpora, and the ranking was then obtained for each text of the OneStopEnglish corpus, at each level.

Resources :

* CEFR dataset introduced in [Xia et al (2016)](https://www.aclweb.org/anthology/W16-0502.pdf)
* The [OneStopEnglish texts](https://www.aclweb.org/anthology/W18-0535.pdf)

## Structure

* [run_analyse_mapping](./run_analyse_mapping.py): Analysis of the levels obtained for each text. Possibility to either plot the histograms per OneStop level, for every level and set of features, or to get simple numbers.
* [run_mapping_onestop](./run_mapping_onestop.py): Storing rankings obtained for each file of the OneStopEnglish corpus.
