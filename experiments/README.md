# Experiments

Code for utilizing the `distribution_inference` package to run experiments.
Example config files are provided in each of the folders to get started.
Folder structure:

- `configs`: Contains sample configuration files for various datasets for attacks, training models, etc.
- `plots`: Folder where generated plots are saved
- `log`: Folder where experimental result (JSON files) are saved

# Setting things up

## Enviornment variables

Please set the environment variable `DDI_DATA_DIRECTORY` to point to a directory where you want to store the datasets. This is used by the `datasets` package to store the datasets, and `DDI_MODELS_DIRECTORY` to point to a directory where models will be trained and saved (and then later loaded for attacks, evaluations).

## Processed Datasets

You can download and extract all the processed datasets (with victim/adversary splits) using [this link](https://archive.org/details/ddi_census_new). You can also selectively download individual datasets (by navigating under `zip files` and downloading the datasets you need).

Make sure your extract all these datasets inside the same dataset directory/folder (`DDI_DATA_DIRECTORY`). For instance, when using celeba and boneage, your structure should look like:

```
DDI_DATA_DIRECTORY
├── celeba
│   ...
├── rsnabone
│   ...
```

After extracting the datasets, please set `DDI_DATA_DIRECTORY` to point to the directory where you unzip the file.

## Training models

`python train_models.py --load_config <your_config_file.json>`

## Launching black-box attacks

`python blackbox_attacks.py --load_config <your_config_file.json> --en <name_for_your_experiment>`

## Launching Permutation-Invariant Network meta-classifier attacks

`python pin.py --load_config your_config_file.json --en name_for_your_experiment`

`python whitebox_pin.py --load_config <your_config_file.json> --en <name_for_your_experiment>`

`python pin_eval.py --load_config your_config_file.json --en name_for_your_experiment`

`python whitebox_attacks_regression.py --load_config <your_config_file.json> --en <name_for_your_experiment>`

`python aga.py --load_config your_config_file.json --en name_for_your_experiment`

`python regression_for_classification.py --load_config <your_config_file.json> --path path_to_saved_meta_classifier`

## Generating plots

`python generate_plots.py --log_path first_result_file second_result_file .... --plot box --dash --legend_titles first_plot second_plot ... --savepath path_to_within_plots_folder`

## Computing n_leaked values

`python get_n_leaked.py --log_path first_result_file second_result_file .... --wanted <select_specific_attack>`

## Use membership-inference to circumvent re-sampling defense

`python mi_attacks.py --load_config <your_config_file.json>`

## Use neighborhood sampling to estimate prediction probabilities (for label-only access)

`python neighboring_attack.py --load_config <your_config_file.json> --en <name_for_your_experiment>`

## Evaluate fairness impact of certain training methods

`python fairness_impact.py --load_config <your_config_file.json>`