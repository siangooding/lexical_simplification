# StanfordNLP parser

-----------------------

I used the StanfordNLP Parser in order to have lemma+POS info for postprocessing GIZA++ output :

* <https://stanfordnlp.github.io/stanfordnlp/installation_usage.html#getting-started>

## Installation

All code was executed using Python 3.7.4 + Ubuntu and a virtual environment.

First run `pip install stanfordnlp` to install the modeul. You will then need to download the models directly from the python terminal

```python
import stanfordnlp
stanfordnlp.download('en')
```

You will be asked whether you want to store it in an alternate directory, you can press yes and change the default directory.

In the [config](./config.yaml) file, change all the paths depending on where you stored the StanfordNLP models.

## Usage

Code was written so that the scripts are executed from the root directory of the whole projet (the root directory contains the `stanfordnlp` folder).

The main script to run is [stanfordnlp_parser.py](./stanfordnlp_parser.py). It takes an input file containing raw text lines, and an output folder to save the output of the StanfordNLP parser formatted in XML format.

Examples of commands to run are provided in the main script of this folder, as well as a toy raw text and its XML StanfordNLP formatting.
