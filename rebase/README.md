# distribution_inference: Package

This package implements various useful functionalities needed in the distributiin-inference pipeline.
Folder structure:

- `attacks`: Implementations for attacks
- `defenses`: Defenses that victim models can deploy (active and passive)
- `config`: Dataclasses for various configurations used in the package.
- `datasets`: Wrappers for various datasets.
- `logging`: Code-logging functionality, used for saving results of experiments.
- `models`: Implementations for models.
- `neff`: Computing n_leaked values.
- `training`: Training models (and some of the meta-classifiers)
- `visualize`: Generating graphs

## Installing the package

Run `pip install -e .` to install the package.

All experiments now use configuration files. Most of the main code- training, testing, attacking models, is now part of a package and is shared between all the datasets. This reduces code redundancy and lowers the chances of introducing bugs when adding new datasets or making edits to existing algorithms.

### Understanding config files

The best way to understand the config files and what each parameters corresponds to, please inspect `config/core.py` in the package. It has detailed docstrings for each of the config classes and their internal parameters.

## Adding your own dataset

To add your own dataset, simply add a new file inside `datasets` corresponding to your dataset. Additionally, you should add your name-dataset mapping to `DATASET_INFO_MAPPING` and `DATASET_WRAPPER_MAPPING` in `datasets/utils.py`. Make sure that your classes extend the ones present in `datasets/base.py`, and implements relevant functions that you want to use.

The function `generate_victim_adversary_splits()` in particular implements functionality to split the data into non-overlapping sets for the victim/adversary each. You can either implement this yourself when creating your dataset, or write it as a function (using your own logic to split, depending on the kind of data) and then call this function.

For example, if you wanted to make splits for the Texas dataset, you would run the following:

```python
from distribution_inference.datasets.utils import get_dataset_information

di = get_dataset_information("texas")()
di.generate_victim_adversary_splits(num_tries=500, split_on_hospitals=False)
```

## Adding your own attack

You can add your own white-box or black-box attack by adding them to the respective folders in `attacks`. Make sure that your classes extend the ones present in `attacks/whitebox/core.py` or `attacks/blackbox/core.py` respectively.

## Adding your own model

You can add your own model by adding them to the `models/core.py` file. Make sure that your model extends from `BaseModel`. When using this model for a particular dataset, you can create a mapping between a model name (string) and the model class itself inside the `get_model()` function of that dataset's `DatasetInformation` class.
