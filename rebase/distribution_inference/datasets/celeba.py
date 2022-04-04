import os
import pandas as pd
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from sklearn.model_selection import train_test_split

import distribution_inference.datasets.base as base
import distribution_inference.datasets.utils as utils
import distribution_inference.models.core as models_core
from distribution_inference.config import TrainConfig, DatasetConfig
from distribution_inference.training.utils import load_model


class DatasetInformation(base.DatasetInformation):
    def __init__(self):
        ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        super().__init__(name="Celeb-A",
                         data_path="celeba",
                         models_path="models_celeba/75_25",
                         properties=["Male", "Young"],
                         values={"Male": ratios, "Young": ratios})
        self.preserve_properties = ['Smiling', 'Young', 'Male', 'Attractive']
        self.supported_properties = [
            '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive',
            'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose',
            'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows',
            'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair',
            'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open',
            'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin',
            'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns',
            'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
            'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace',
            'Wearing_Necktie', 'Young'
        ]

    def get_model(self, parallel: bool = False, fake_relu: bool = False,
                  latent_focus=None, is_large: bool = False,
                  cpu: bool = False) -> nn.Module:
        if is_large:
            model = models_core.InceptionModel(
                fake_relu=fake_relu,
                latent_focus=latent_focus)
        else:
            model = models_core.MyAlexNet(
                fake_relu=fake_relu,
                latent_focus=latent_focus)
        if not cpu:
            model = model.cuda()
        if parallel:
            model = nn.DataParallel(model)
        return model

    def _victim_adv_identity_split(self, identities, attrs,
                                   n_tries: int = 1000, adv_ratio=0.25):
        # Create mapping of IDs to face images
        mapping = {}
        for i, id_ in enumerate(identities):
            mapping[id_] = mapping.get(id_, []) + [i]

        # Group-check for attribute values
        def get_vec(spl):
            picked_keys = np.array(list(mapping.keys()))[spl]
            collected_ids = np.concatenate([mapping[x] for x in picked_keys])
            vals = [attrs[pp].iloc[collected_ids].mean()
                    for pp in self.self.preserve_properties]
            return np.array(vals)

        # Take note of original ratios
        ratios = np.array([attrs[pp].mean() for pp in self.self.preserve_properties])

        iterator = tqdm(range(n_tries))
        best_splits = None, None
        best_diff = (np.inf, np.inf)
        for _ in iterator:
            # Generate random victim/adv split
            randperm = np.random.permutation(len(mapping))
            split_point = int(len(randperm) * adv_ratio)
            adv_ids, victim_ids = randperm[:split_point], randperm[split_point:]

            # Get ratios for these splits
            vec_adv = get_vec(adv_ids)
            vec_victim = get_vec(victim_ids)

            # Measure ratios for images contained in these splits
            diff_adv = np.linalg.norm(vec_adv-ratios)
            diff_victim = np.linalg.norm(vec_victim-ratios)

            if best_diff[0] + best_diff[1] > diff_adv + diff_victim:
                best_diff = (diff_adv, diff_victim)
                best_splits = adv_ids, victim_ids

            iterator.set_description(
                "Best ratio differences: %.4f, %.4f" % (best_diff[0], best_diff[1]))

        # Extract indices corresponding to splits
        split_adv, split_victim = best_splits

        picked_keys_adv = np.array(list(mapping.keys()))[split_adv]
        adv_mask = np.concatenate([mapping[x] for x in picked_keys_adv])

        picked_keys_victim = np.array(list(mapping.keys()))[split_victim]
        victim_mask = np.concatenate([mapping[x] for x in picked_keys_victim])

        return adv_mask, victim_mask

    def generate_victim_adversary_splits(self,
                                         adv_ratio: float = 0.25,
                                         test_ratio=None,
                                         num_tries: int = 5000):
        """
            Generate and store data offline for victim and adversary
            using the given dataset. Use this method only once for the
            same set of experiments.
        """
        def get_identities():
            fpath = os.path.join(self.base_data_dir, "identity_CelebA.txt")
            identity = pd.read_csv(fpath, delim_whitespace=True,
                                   header=None, index_col=0)
            return np.array(identity.values).squeeze(1)

        def get_splits():
            fpath = os.path.join(self.base_data_dir, "list_eval_partition.txt")
            splits = pd.read_csv(fpath, delim_whitespace=True,
                                 header=None, index_col=0)
            return splits

        # Load metadata files
        splits = get_splits()
        ids = get_identities()
        attrs, _ = _get_attributes(self.base_data_dir)
        filenames = np.array(splits.index.tolist())

        # 0 train, 1 validation, 2 test
        train_mask = np.logical_or(splits[1].values == 0, splits[1].values == 1)
        test_mask = splits[1].values == 2

        # Splits on test data
        test_adv, test_victim = self._victim_adv_identity_split(
            ids[test_mask], attrs[test_mask],
            n_tries=num_tries, adv_ratio=adv_ratio)
        mask_locs = np.nonzero(test_mask)[0]
        test_adv_filenames = filenames[mask_locs[test_adv]]
        test_victim_filenames = filenames[mask_locs[test_victim]]

        # Splits on train data
        train_adv, train_victim = self._victim_adv_identity_split(
            ids[train_mask], attrs[train_mask],
            n_tries=num_tries, adv_ratio=adv_ratio)
        mask_locs = np.nonzero(train_mask)[0]
        train_adv_filenames = filenames[mask_locs[train_adv]]
        train_victim_filenames = filenames[mask_locs[train_victim]]

        # Save split files for later use
        def save(data, path):
            with open(os.path.join(self.base_data_dir, path), 'w') as f:
                f.writelines("%s\n" % l for l in data)

        save(test_adv_filenames, os.path.join(
            "splits", "75_25", "adv", "test.txt"))
        save(test_victim_filenames, os.path.join(
            "splits", "75_25", "victim", "test.txt"))
        save(train_adv_filenames, os.path.join(
            "splits", "75_25", "adv", "train.txt"))
        save(train_victim_filenames, os.path.join(
            "splits", "75_25", "victim", "train.txt"))


class CelebACustomBinary(base.CustomDataset):
    def __init__(self, classify, filelist_path, attr_dict,
                 prop, ratio, cwise_sample,
                 shuffle=False, transform=None):
        self.attr_dict = attr_dict
        self.transform = transform
        self.info_object = DatasetInformation()
        self.classify_index = self.info_object.supported_properties.index(classify)

        # Get filenames
        with open(filelist_path) as f:
            self.filenames = f.read().splitlines()

        # Sort to get deterministic order
        self.filenames.sort()

        # Apply requested filter
        self.filenames = self._ratio_sample_data(
            self.filenames, self.attr_dict,
            classify, prop, ratio, cwise_sample)

        if shuffle:
            np.random.shuffle(self.filenames)

        self.num_samples = len(self.filenames)

    def _create_df(self, attr_dict, filenames):
        # Create DF from filenames to use heuristic for ratio-preserving splits
        all = []
        for filename in filenames:
            y = list(attr_dict[filename].values())
            all.append(y + [filename])
        df = pd.DataFrame(data=all,
                          columns=self.info_object.supported_properties + ['filename'])
        return df

    def _ratio_sample_data(self,
                           filenames, attr_dict, label_name,
                           prop, ratio, cwise_sample):
        # Make DF
        df = self._create_df(attr_dict, filenames)

        # Make filter
        def condition(x): return x[prop] == 1

        parsed_df = utils.heuristic(
                        df, condition, ratio,
                        cwise_sample, class_imbalance=1.0,
                        n_tries=100, class_col=label_name,
                        verbose=True)
        # Extract filenames from parsed DF
        return parsed_df["filename"].tolist()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        x = Image.open(os.path.join(
            self.info_object.base_data_dir, "img_align_celeba", filename))
        y = np.array(list(self.attr_dict[filename].values()))

        if self.transform:
            x = self.transform(x)

        return x, y[self.classify_index], y


class CelebaWrapper(base.CustomDatasetWrapper):
    def __init__(self, data_config: DatasetConfig, skip_data: bool = False):
        super().__init__(data_config, skip_data)
        self.info_object = DatasetInformation()

        # Make sure specified label is valid
        if self.classify not in self.info_object.preserve_properties:
            raise ValueError("Specified label not available for images")

        train_transforms = [
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ]
        self.test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])

        if self.augment:
            augment_transforms = [
                transforms.RandomAffine(degrees=20,
                                        translate=(0.2, 0.2),
                                        shear=0.2),
                transforms.RandomHorizontalFlip()
            ]
            train_transforms = augment_transforms + train_transforms
        self.train_transforms = transforms.Compose(train_transforms)

    def load_data(self):
        # Read attributes file to get attribute names
        attrs, _ = _get_attributes(self.info_object.base_data_dir)
        # Create mapping between filename and attributes
        attr_dict = attrs.to_dict(orient='index')

        # Use relevant file split information
        filelist_train = os.path.join(
            self.info_object.base_data_dir,
            "splits", "75_25", self.split, "train.txt")
        filelist_test = os.path.join(
            self.info_object.base_data_dir,
            "splits", "75_25", self.split, "test.txt")

        # Define number of sub-samples
        prop_wise_subsample_sizes = {
            "Smiling": {
                "adv": {
                    "Male": (10000, 1000),
                    "Attractive": (10000, 1200),
                    "Young": (6000, 600)
                },
                "victim": {
                    "Male": (15000, 3000),
                    "Attractive": (30000, 4000),
                    "Young": (15000, 2000)
                }
            },
            "Male": {
                "adv": {
                    "Young": (3000, 350),
                },
                "victim": {
                    "Young": (8000, 1400),
                }
            }
        }

        cwise_sample = prop_wise_subsample_sizes[self.classify][self.split][self.prop]
        if self.cwise_samples is not None:
            self.cwise_sample = self.cwise_samples

        # Create datasets
        ds_train = CelebACustomBinary(
            self.classify, filelist_train, attr_dict,
            self.prop, self.ratio, cwise_sample[0],
            transform=self.train_transforms,)
        ds_val = CelebACustomBinary(
            self.classify, filelist_test, attr_dict,
            self.prop, self.ratio, cwise_sample[1],
            transform=self.test_transforms)
        return ds_train, ds_val

    def get_loaders(self, batch_size, shuffle=True,
                    eval_shuffle=False, val_factor=2,
                    num_workers=16, prefetch_factor=20):
        self.ds_train, self.ds_val = self.load_data()
        return super().get_loaders(batch_size, shuffle=shuffle,
                                   eval_shuffle=eval_shuffle,
                                   val_factor=val_factor,
                                   num_workers=num_workers,
                                   prefetch_factor=prefetch_factor)

    def get_save_dir(self, train_config: TrainConfig) -> str:
        base_models_dir = self.info_object.base_models_dir
        subfolder_prefix = os.path.join(
            self.split, self.prop, str(self.ratio)
        )

        if train_config.misc_config and train_config.misc_config.adv_config:
            # Extract epsilon to be used
            adv_folder_prefix = "adv_train_"
            adv_config = train_config.misc_config.adv_config
            if adv_config.scale_by_255:
                # Use 'int' value
                epsilon_val = int(adv_config.epsilon * 255)
                adv_folder_prefix += ("%d" % epsilon_val)
            else:
                # If actual epsilon value, use as it is
                epsilon_val = adv_config.epsilon
                adv_folder_prefix += ("%.4f" % epsilon_val)
            subfolder_prefix = os.path.join(
                subfolder_prefix, adv_folder_prefix)

        save_path = os.path.join(base_models_dir, subfolder_prefix)

        # # Make sure this directory exists
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        return save_path

    def load_model(self, path: str, on_cpu: bool = False) -> nn.Module:
        info_object = DatasetInformation()
        model = info_object.get_model(cpu=on_cpu)
        return load_model(model, path)


def _get_attributes(base_data_dir):
    fpath = os.path.join(base_data_dir, "list_attr_celeba.txt")
    attrs = pd.read_csv(fpath, delim_whitespace=True, header=1)
    attrs = (attrs + 1) // 2
    attr_names = list(attrs.columns)
    return attrs, attr_names
