# Experiments

Code for utilizing the `distribution_inference` package to run experiments.
Example config files are provided in each of the folders to get started.
Folder structure:

- `configs`: Contains sample configuration files for various datasets for attacks, training models, etc.
- `plots`: Folder where generated plots are saved
- `log`: Folder where experimental result (JSON files) are saved

# Models

## Training

`python train_models.py --load_config your_config_file.json`

## Evaluation

`python task_eval.py --load_config your_config_file.json`

## KL Divergence Attack (KL) and other Black-Box attacks

`python blackbox_attacks.py --load_config your_config_file.json --en name_for_your_experiment`

## Permutation Invariant Network (PIN)

### Train attack

`python pin.py --load_config your_config_file.json --en name_for_your_experiment`

## Evaluate attack

`python pin_eval.py --load_config your_config_file.json --en name_for_your_experiment`

## Affinity Graph Attack (AGA)

`python aga.py --load_config your_config_file.json --en name_for_your_experiment`

## Evaluate attack

`python aga_eval.py --load_config your_config_file.json --en name_for_your_experiment`

## Combined Attack (AGA+KL)

`python agakl.py --load_config your_config_file.json --en name_for_your_experiment`

## Analyse impact of re-sampling defense on fairness

`python fairness_impact.py --load_config your_config_file.json`

## Compute n_leaked values for a set of results

`python get_nleaked.py --load_config your_config_file.json`

## Generating plots

`python generate_plots.py --log_path first_result_file second_result_file .... --legend_titles first_plot second_plot ... --savepath path_to_save_plot`
