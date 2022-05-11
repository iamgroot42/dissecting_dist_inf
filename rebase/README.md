# distribution_inference: Package

### Installing the package

Run `pip install -e .` to install the package.

All experiments now use configuration files. Most of the main code- training, testing, attacking models, is now part of a package and is shared between all the datasets. This reduces code redundancy and lowers the chances of introducing bugs when adding new datasets or making edits to existing algorithms.

### Adding your own dataset

To add your own dataset, simply add a new file inside `datasets` corresponding to your dataset. Additionally, you should add your name-dataset mapping to `DATASET_INFO_MAPPING` and `DATASET_WRAPPER_MAPPING` in `datasets/utils.py`. Make sure that your classes extend the ones present in `datasets/base.py`

### Adding your own attack

You can add your own white-box or black-box attack by adding them to the respective folders in `attacks`. Make sure that your classes extend the ones present in `attacks/whitebox/core.py` or `attacks/blackbox/core.py` respectively.