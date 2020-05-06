# Distribution of word complexities

-----------------------

I wanted to explore how the complexities of the words were distributed across levelled corpora. I retrieved word complexities at sentence levels and text levels, and plotted histograms with the data I collected. All words complexities computed can be found in the [distrib](./distrib.pickle) pickle file.

Resources :

* CEFR dataset introduced in [Xia et al (2016)](https://www.aclweb.org/anthology/W16-0502.pdf)

## Structure

* [distrib_level_vis](./distrib_level_vis.py): Distribution levels visualization
* [distrib_level](./distrib_level.py): Storing complexities for each token within a text/sentence into a pickle file
* [distrib](./distrib.pickle): collected data
