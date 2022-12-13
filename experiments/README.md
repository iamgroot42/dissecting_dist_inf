# Experiments

Code for utilizing the `distribution_inference` package to run experiments.
Folder structure:

- `configs`: Contains sample configuration files for various datasets for attacks, training models, etc.
- `plots`: Folder where generated plots are saved
- `log`: Folder where experimental result (JSON files) are saved

## Training models

`python train_models.py --load_config <your_config_file.json>`

## Launching black-box attacks

`python blackbox_attacks.py --load_config <your_config_file.json> --en <name_for_your_experiment>`

## Launching Permutation-Invariant Network meta-classifier attacks

### For binary classification

`python whitebox_pin.py --load_config <your_config_file.json> --en <name_for_your_experiment>`

### For direct regression

`python whitebox_attacks_regression.py --load_config <your_config_file.json> --en <name_for_your_experiment>`

### For using regression variant (saved) to perform binary classification

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