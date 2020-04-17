# How to use GIZA++

-----------------------

In order to see how words were simplified from a higher level text to a lower level one, I used GIZA++ to first align sentences, and then used the StanfordNLP Parser for post-processing and selecting specific candidates.

Resources :

* GIZA++ (v. 1.0.3) : <http://www.statmt.org/moses/giza/GIZA++.html>

* Compiling GIZA++ : <https://medium.com/@weGotlieb/using-giza-for-bilingual-sentence-alignment-8b2dc828a04d>

* StanfordNLP Parser (see [README](./stanfordnlp/README.md) for usage of StanfordNLP parser) : <https://stanfordnlp.github.io/stanfordnlp/installation_usage.html#getting-started>

* Post-processing of GIZA++ output : Recursive Context-Aware Lexical Simplification - Gooding and Kochmar - 2019

* OneStopEnglish aligned sentences : <https://github.com/nishkalavallabhi/OneStopEnglishCorpus/tree/master/Sentence-Aligned>

## Installation

All code was executed using Python 3.7.4 + Ubuntu and a virtual environment.

First install GIZA++ and make it work properly on your machine. I followed instructions from <https://medium.com/@weGotlieb/using-giza-for-bilingual-sentence-alignment-8b2dc828a04d> and it proved very useful.

Install the StanfordNLP parser : follow indications given in [README](./stanfordnlp/README.md) dedicated to the parser.

## Usage

The command line below are provided when aligning words between aligned sentences from the Advanced level (higher level) and the Elementary level (lower level), in the OneStopEnglish corpus. The initial file containing the aligned sentences - `ADV-ELE.txt` -, is in the `./onestop/aligned` folder, whereas I stored all info needed pre-GIZA++ in the `./giza-pre-process` folder, and all the info relative to GIZA++ output in the `./giza-output` folder. All commands given are assumed to be run from the `exp_alignment` folder from the root directory of the project.

* **Creating folders.** Once you have GIZA++ installed you should have a `giza-pp-master` folder. Create two additional folders in root directory of main project, `giza-output` and `giza-pre-process`. In both folders, create three additional folders : `adv-int`, `adv-ele` and `int-ele`.

* **Pre-processing the aligned sentences.** The original aligned sentences contained some no encoding-friendly characters hence I cleaned the text manually. The cleaned versions of the aligned sentences are in the `./onestop/aligned` folder, namely `ADV-ELE.txt`, `ADV-INT.txt` and `ELE-INT.txt`. It can be reused directly.

* **Create two separate files for each level.** In order to use GIZA++, two files need to be created, with one sentence per line and separated levels.

```bash
python ./onestop/aligned/create_sep_file.py -i ./onestop/aligned/ADV-ELE.txt -o ./giza-pre-process/adv-ele/
```

* **Pre-processing the text for GIZA++ input.** The aim of this step is to tokenize the text for GIZA++ input.

```bash
./mosesdecoder/scripts/tokenizer/tokenizer.perl -l en < ./giza-pre-process/adv-ele/onestop.en-en.ele > ./giza-pre-process/adv-ele/onestop.en-en.tok.ele
./mosesdecoder/scripts/tokenizer/tokenizer.perl -l en < ./giza-pre-process/adv-ele/onestop.en-en.adv > ./giza-pre-process/adv-ele/onestop.en-en.tok.adv
```

* **Creating three types of files to run GIZA++.** .vcb with vocabulary words for each of the two languages + .snt sentence file + .cooc concurrence file. First command to create .snt and .vsb file, second to create .cooc file.

```bash
./giza-pp-master/GIZA++-v2/plain2snt.out ./giza-pre-process/adv-ele/onestop.en-en.tok.adv ./giza-pre-process/adv-ele/onestop.en-en.tok.ele
./giza-pp-master/GIZA++-v2/snt2cooc.out ./giza-pre-process/adv-ele/onestop.en-en.tok.adv.vcb ./giza-pre-process/adv-ele/onestop.en-en.tok.ele.vcb  ./giza-pre-process/adv-ele/onestop.en-en.tok.adv_onestop.en-en.tok.ele.snt > ./giza-pre-process/adv-ele/corp-adv-ele.cooc
```

* **Running GIZA++.** Aligning words between paired sentences.

```bash
./giza-pp-master/GIZA++-v2/GIZA++ -S ./giza-pre-process/adv-ele/onestop.en-en.tok.adv.vcb -T ./giza-pre-process/adv-ele/onestop.en-en.tok.ele.vcb -C ./giza-pre-process/adv-ele/onestop.en-en.tok.adv_onestop.en-en.tok.ele.snt -CoocurrenceFile ./giza-pre-process/adv-ele/corp-adv-ele.cooc -outputpath ./giza-output/
```

* **Getting POS and lemmas of each token in sentence with the StanfordNLP parser.** For post-processing GIZA++ output, I needed to have POS and lemma information on each token. I chose StanfordNLP parser as it provided the option to enter tokens as input rather than raw sentence.

```bash
python ../stanfordnlp_/stanfordnlp_parser.py -i ./giza-pre-process/adv-ele/onestop.en-en.tok.adv -o ./giza-pre-process/adv-ele/
python ../stanfordnlp_/stanfordnlp_parser.py -i ./giza-pre-process/adv-ele/onestop.en-en.tok.ele -o ./giza-pre-process/adv-ele/
```

* **Save selected word pairs as .pkl file.** Analyse GIZA++ output in parallel with the POS-tagged and lemmatized paired sentences and store final candidates in .pkl file. The .pkl file contains a dictionary : key = higher level word, values = lower level correspondences. Note that depending on the time you ran the GIZA++ final script, the name of the giza .VA3.file might change.

```bash
python ./giza-output/find_candidates.py -g ./giza-output/2020-03-20.110444.ib431.VA3.final -so ./giza-pre-process/adv-ele/onestop.en-en.tok.adv.xml.stanfordnlp -t ./giza-pre-process/adv-ele/onestop.en-en.tok.ele.xml.stanfordnlp -sa ./onestop/aligned/adv-ele.pkl
```
