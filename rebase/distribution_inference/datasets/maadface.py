"""
    Currently limited to only adjusting ratio of males/females
"""
import os
from distribution_inference.defenses.active.shuffle import ShuffleDefense
import pandas as pd
from torchvision import transforms
from PIL import Image
import torch as ch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from typing import  List

import distribution_inference.datasets.base as base
import distribution_inference.datasets.utils as utils
import distribution_inference.models.contrastive as models_contrastive
from distribution_inference.config import TrainConfig, DatasetConfig
from distribution_inference.training.utils import load_model
from distribution_inference.utils import model_compile_supported
import pickle


# TODO:
# <DONE> 1. Add support for exclusion of certain people (list/singled out) based on input
# <DONE> 2. Find SOTA contrastive training methods (and models) to implement for feature extractor
# <DONE> 3. Define data splits with a) victim's set of people b) adv's set of people
# 4. Collect images for target people from the internet in post-dataset era to have no overlap in images


class DatasetInformation(base.DatasetInformation):
    def __init__(self, epoch_wise: bool = False):
        ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        super().__init__(name="MAADFaceMale",
                         data_path="vggface2",
                         models_path="models_maadface/75_25_nobalancing",
                         properties=['Male'],
                         values={"Male": ratios},
                         supported_models=["arcface_resnet"],
                         default_model="arcface_resnet",
                         epoch_wise=epoch_wise)
        # At this point, only focus on Male since it is the only property
        # that is consistent at the class level (same person would have same attribute label throughout)
        # Could consider others too, but re-sampling to balance out this property would require
        # adjustments in the class label distributions, which might turn out to be a secondary source of leakage
        # Plus, for some reason, other interesting attributes like "Race" are somehow missing for many cases?
        self._populate_attrs()

    def _populate_attrs(self):
        self.maad_face_mapping = pd.read_pickle(os.path.join(self.base_data_dir, "MAAD_Face_1.0.pkl"))
        # Filter out instances where requested properties have label 0
        # (i.e. not present)
        for pp in self.properties:
            if len(self.maad_face_mapping[self.maad_face_mapping[pp] == 0]) != 0:
                raise ValueError(f"WARNING: {pp} has {len( self.maad_face_mapping[self.maad_face_mapping[pp] == 0])} instances with label 0")
            # Replace -1 with 0
            self.maad_face_mapping[pp] = self.maad_face_mapping[pp].replace(-1, 0)
        # From this, create a Name-prop mapping
        if os.path.exists(os.path.join(self.base_data_dir, "MAAD_Face_1.0_name_prop_mapping.pkl")):
            with open(os.path.join(self.base_data_dir, "MAAD_Face_1.0_name_prop_mapping.pkl"), 'rb') as handle:
                self.name_prop_mapping = pickle.load(handle)
        else:
            self.name_prop_mapping = {}
            for idx, row in tqdm(self.maad_face_mapping.iterrows(), desc="Creating name-prop mapping", total=len(self.maad_face_mapping)):
                name = row["Filename"].split("/")[0]
                if name not in self.name_prop_mapping:
                    self.name_prop_mapping[name] = {}
                for pp in self.properties:
                    self.name_prop_mapping[name][pp] = row[pp]
            # Save this in pickle file (for faster loading)
            with open(os.path.join(self.base_data_dir, "MAAD_Face_1.0_name_prop_mapping.pkl"), 'wb') as handle:
                pickle.dump(self.name_prop_mapping, handle)

    def get_model(self, parallel: bool = False, fake_relu: bool = False,
                  latent_focus=None, cpu: bool = False,
                  model_arch: str = None,
                  for_training: bool = False,
                  n_people: int = None) -> nn.Module:
        if model_arch is None or model_arch == "None":
            model_arch = self.default_model

        if model_arch == "arcface_resnet":
            model = models_contrastive.ArcFaceResnet(n_people=n_people)
        else:
            raise NotImplementedError("Model architecture not supported")

        if parallel:
            model = nn.DataParallel(model)
        if not cpu:
            model = model.cuda()
        
        if for_training and model_compile_supported():
            model = ch.compile(model)

        return model

    def get_split_save_path(self, split: str, train: bool) -> str:
        if split not in ["victim", "adv"]:
            raise ValueError("Invalid split specified!")
        suffix = "_train_" if train else "_test_"
        return os.path.join(self.base_data_dir, "%s%snames.txt" % (split, suffix))

    def generate_victim_adversary_splits(self,
                                         adv_ratio: float = 0.25,
                                         test_ratio=None,
                                         num_tries: int = None):
        """
            Generate and store data offline for victim and adversary
            using the given dataset. Use this method only once for the
            same set of experiments.
        """
        # TODO: Generate test/train splits of images here itself (for reproducing results across machines)
        directory = self.base_data_dir
        # Open file named 'test_names.txt' and read the names of the test people
        with open(os.path.join(directory, "test_names.txt"), "r") as f:
            test_names = f.readlines()
        # Do same thing for train names
        with open(os.path.join(directory, "train_names.txt"), "r") as f:
            train_names = f.readlines()
        # Remove the newline character from the end of each name
        all_names = [("train", name.strip()) for name in train_names]
        all_names += [("test", name.strip()) for name in test_names]

        # Randomly split into victim and adv based on adv_ratio
        np.random.shuffle(all_names)
        adv_names = all_names[:int(len(all_names) * adv_ratio)]
        victim_names = all_names[int(len(all_names) * adv_ratio):]

        def make_train_test_splits(entry):
            files_within = os.listdir(os.path.join(directory, entry[0], entry[1]))
            # Shuffle these files
            np.random.shuffle(files_within)
            # Split into train and test
            num_test = int(len(files_within) * test_ratio)
            test_file_paths = files_within[:num_test]
            train_files_paths = files_within[num_test:]
            return train_files_paths, test_file_paths

        # Save split files for later use
        def save(data, path):
            with open(os.path.join(self.base_data_dir, path), 'w') as f:
                for line in data:
                    f.write(",".join(line) + "\n")

        # Make train/test splits for adv and victim
        adv_paths = map(make_train_test_splits, tqdm(adv_names, desc="Making adversary train/test splits"))
        combined_paths = list(map(lambda x, y: (list(x) + y[0], list(x) + y[1]), adv_names, adv_paths))
        adv_train_paths = list(map(lambda x: x[0], combined_paths))
        adv_test_paths = list(map(lambda x: x[1], combined_paths))

        # Format saved as :<train/test>,<name>,<filenames>
        save(adv_train_paths, self.get_split_save_path("adv", train=True))
        save(adv_test_paths, self.get_split_save_path("adv", train=False))

        # Now move over to victim splits
        victim_paths = map(make_train_test_splits, tqdm(victim_names, desc="Making adversary train/test splits"))
        combined_paths = list(map(lambda x, y: (list(x) + y[0], list(x) + y[1]), victim_names, victim_paths))
        victim_train_paths = list(map(lambda x: x[0], combined_paths))
        victim_test_paths = list(map(lambda x: x[1], combined_paths))

        save(victim_train_paths, self.get_split_save_path("victim", train=True))
        save(victim_test_paths, self.get_split_save_path("victim", train=False))


class MAADFaceDataset(base.CustomDataset):
    def __init__(self,
                 data_folder: str,
                 class_to_idx: dict,
                 files_list: List[str],
                 people_selected,
                 name_to_requested_prop_mapping,
                 prop, shuffle: bool = False,
                 transform=None):
        super().__init__()
        # TODO: Add support for people present/absent, or even groups of people
        self.transform = transform

        # Load names of people
        id_mapping = {}
        for record in files_list:
            items = record.split(',')
            prefix, id = items[:2]
            files_to_use = items[2:]
            id_mapping[id] = list(map(lambda x: os.path.join(data_folder, prefix, id, x), files_to_use))

        # Train split already identified set of people, just need to get their test images
        self.people_selected = people_selected
    
        # Flatten out filenames, store corresponding labels
        self.filenames, self.labels, self.prop_labels = [], [], []
        for id in self.people_selected:
            images = id_mapping[id]
            self.filenames += images
            num_people = len(id_mapping[id])
            self.labels += [id] * num_people
            self.prop_labels += [name_to_requested_prop_mapping[id]
                                 [prop]] * num_people
        
        # Create name-to-label mapping
        self.classes = list(class_to_idx.keys())
        self.labels = [class_to_idx[l] for l in self.labels]

        self.filenames = np.array(self.filenames)
        self.labels = np.array(self.labels)
        self.prop_labels = np.array(self.prop_labels)

        if shuffle:
            # Create shuffling order and apply same shuffling to labels and data
            shuffling_order = np.arange(len(self.filenames))
            np.random.shuffle(shuffling_order)
            self.filenames = self.filenames[shuffling_order]
            self.labels = self.labels[shuffling_order]

        self.num_samples = len(self.filenames)

    def __getitem__(self, idx):
        # Open image
        filename = self.filenames[idx]
        x = Image.open(filename)
        if self.transform:
            x = self.transform(x)

        y = self.labels[idx] # Person identified
        prop_label = self.prop_labels[idx] # Property label ID

        return x, y, prop_label


class MaadFaceWrapper(base.CustomDatasetWrapper):
    def __init__(self,
                 data_config: DatasetConfig,
                 skip_data: bool = False,
                 label_noise: float = 0,
                 epoch: bool = False,
                 shuffle_defense: ShuffleDefense = None,):
        super().__init__(data_config,
                         skip_data=skip_data,
                         label_noise=label_noise,
                         shuffle_defense=shuffle_defense)
        self.info_object = DatasetInformation(epoch_wise=epoch)

        train_transforms = [
            transforms.Resize([128, 128]),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ]
        self.test_transforms = transforms.Compose([
            transforms.Resize([128, 128]),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])

        if self.augment:
            # Add more augmentations?
            augment_transforms = [
                transforms.RandomHorizontalFlip()
            ]
            train_transforms = augment_transforms + train_transforms
        self.train_transforms = transforms.Compose(train_transforms)

        # Define (number of people to pick, number of test images per person)
        self._prop_wise_subsample_sizes = {
            "Male": {
                "adv": (625, 30),
                "victim": (2500, 50)
            }
        }
        self.n_people, self.n_in_test = self._prop_wise_subsample_sizes[self.prop][self.split]
    
    def _create_df(self, id_list, name_to_requested_prop_mapping, prop):
        # Base assumption: roughly same number of people in each class
        all = []
        for id in id_list:
            prop_label = name_to_requested_prop_mapping[id][prop]
            all.append([id, prop_label])
        df = pd.DataFrame(data=all, columns=['id', prop])
        return df
    
    def _ratio_sample_data(self,
                           id_list, name_to_requested_prop_mapping, prop,
                           ratio, num_people_total):
        # Make DF
        df = self._create_df(id_list, name_to_requested_prop_mapping, prop)

        # Make filter
        def condition(x): return x[prop] == 1

        parsed_df = utils.multiclass_heuristic(
            df, condition, ratio,
            total_samples=num_people_total,
            n_tries=10,
            class_ratio_maintain=False,
            verbose=True)

        # Get people shortlisted
        return parsed_df["id"].tolist()
    
    def subsample_people(self, files, total_people):
        # Extract id_list from files
        id_list = list(map(lambda x: x.split(',')[1], files))
        name_to_requested_prop_mapping = self.info_object.name_prop_mapping

        # Apply requested filter at the level of people (not images)
        people_selected = self._ratio_sample_data(
            id_list, name_to_requested_prop_mapping, self.prop,
            self.ratio, total_people)
        return people_selected

    def load_data(self):
        # Use relevant file split information
        with open(self.info_object.get_split_save_path(self.split, train=True), 'r') as f:
            files_train = list(map(lambda x: x.rstrip(), f.readlines()))
        
        with open(self.info_object.get_split_save_path(self.split, train=False), 'r') as f:
            files_test = list(map(lambda x: x.rstrip(), f.readlines()))
        
        self.people_selected = self.subsample_people(files_train, self.n_people)
        name_to_requested_prop_mapping = self.info_object.name_prop_mapping
        # Create name to label mapping
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.people_selected)}
        
        data_folder = self.info_object.base_data_dir
        ds_train = MAADFaceDataset(
            data_folder,
            self.class_to_idx,
            files_train,
            self.people_selected,
            name_to_requested_prop_mapping,
            self.prop,
            transform=self.train_transforms,
            shuffle=True)
        ds_val = MAADFaceDataset(
            data_folder,
            self.class_to_idx,
            files_test,
            self.people_selected,
            name_to_requested_prop_mapping,
            self.prop,
            transform=self.test_transforms,
            shuffle=False)
        return ds_train, ds_val

    def get_loaders(self, batch_size: int,
                    shuffle: bool = True,
                    eval_shuffle: bool = False,
                    val_factor: int = 2,
                    num_workers: int = 4,
                    prefetch_factor: int = 2,
                    indexed_data=None):
        self.ds_train, self.ds_val = self.load_data()

        return super().get_loaders(batch_size, shuffle=shuffle,
                                   eval_shuffle=eval_shuffle,
                                   val_factor=val_factor,
                                   num_workers=num_workers,
                                   prefetch_factor=prefetch_factor,
                                   train_sampler=None)

    def get_save_dir(self, train_config: TrainConfig, model_arch: str) -> str:
        base_models_dir = self.info_object.base_models_dir
        subfolder_prefix = os.path.join(
            self.split, self.prop, str(self.ratio)
        )
        if not (train_config.misc_config and train_config.misc_config.contrastive_config):
            raise ValueError("Only contrastive training is supported for this dataset")
        else:
            contrastive_config = train_config.misc_config.contrastive_config

        # Standard logic
        if model_arch == "None" or model_arch is None:
            model_arch = self.info_object.default_model
        if model_arch not in self.info_object.supported_models:
            raise ValueError(f"Model architecture {model_arch} not supported")

        # Check if augmented version or not
        if train_config.data_config.augment:
            model_arch += "_aug"
        base_models_dir = os.path.join(base_models_dir, model_arch)

        save_path = os.path.join(base_models_dir, subfolder_prefix)

        # Make sure this directory exists
        if not os.path.isdir(save_path):
            os.makedirs(save_path, exist_ok=True)

        return save_path

    def load_model(self, path: str,
                   on_cpu: bool = False,
                   model_arch: str = None) -> nn.Module:
        model = self.info_object.get_model(
            cpu=on_cpu, model_arch=model_arch, n_people=self.n_people)
        return load_model(model, path, on_cpu=on_cpu)
