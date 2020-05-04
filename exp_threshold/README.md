# The impact of threshold on the lexical simplification system

-----------------------

I wanted to explore the impact of threshold on the lexical simplification system introduced by [Gooding and Kochmar (2019)](https://www.aclweb.org/anthology/D19-1491.pdf). Hence I used 5 original advanced texts and ran the system with two different thresholds: 0.5 and 0.8.

Resources :

* _Amazon_, _Amsterdam_, _Anita_, _Arctic mapping_, _Banksy_ advanced texts of the [OneStopEnglish corpus](https://www.aclweb.org/anthology/W18-0535.pdf).

## Structure

* [recursive_config_threshold](./recursive_config_threshold.yaml): configuration file for the lexical simplification system. Should not be changed for reproducibility.
* [run_exp_threshold](./run_exp_threshold.py): Main script to execute for this experiment. Ensure that texts were pre added in the correponding `TXT_PATHS` variable.
