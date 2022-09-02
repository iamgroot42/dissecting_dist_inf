# Dissecting Distribution Inference

Code for our paper 'Dissecting Distribution Inference'. Majority of our code is setup as a library, containing implementations of attacks, defenses, model training, visualization, and other useful utilities
Adding attacks or datasets is easy, and requires inhereting from the appropriate base class with minimal changes.

Folder structure:

- `rebase` : Main package that includes functionality for training models, launching white-box and black-box attacks, and logging & generating plots.
- `experiments` : Contains scripts to use the package for experiments.

