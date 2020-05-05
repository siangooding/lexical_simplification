# Final main experiment

-----------------------

The objective of the main experiment was to see how the readability classifier was performing with original different advanced texts simplified at different levels. I consequently used my modified lexical simplification system, initally taken from [Gooding and Kochmar (2019)](https://www.aclweb.org/anthology/D19-1491.pdf) to first simplify my texts at two differents levels: KET (level 2) and FCE (level 4). I then used to readability code introduced in [Xia et al (2016)](https://www.aclweb.org/anthology/W16-0502.pdf).

## Main steps

I compartmentalized my main experiment into several sub experiments:

* Storing texts simplified with different parameters
* Storing features for classifier of those simplified texts
* Cross-validation with frozen classification parameters
* Cross-validation with frozen recursive simplification parameters

## Structure

* [format_config](./format_config.yaml): configuration file for formatting texts for the readability classifier
* [helpers](./helpers.py):
* [recursive_config_exp](./recursive_config_exp.yaml): configuration file for the recursive simplification system
* [run_cv_frozen_classification](./run_cv_frozen_classification.py): CV to find the best set of parameters for the simplification system
* [run_cv_frozen_ls](./run_cv_frozen_ls.py): CV to find the best set of parameters for the readability system
* [run_format_text](./run_format_text.py): Formatting texts for the input classifier (xml + rasp output)
* [run_pre_cache_features](./run_pre_cache_features.py): Storing features for the classifier
* [run_store_texts](./run_store_texts.py): Simplifying texts
