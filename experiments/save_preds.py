from simple_parsing import ArgumentParser
from pathlib import Path
import os
import pickle
from distribution_inference.datasets.utils import get_dataset_wrapper, get_dataset_information
from distribution_inference.attacks.blackbox.utils import get_attack, calculate_accuracies, get_vic_adv_preds_on_distr
from distribution_inference.attacks.blackbox.core import PredictionsOnDistributions
from distribution_inference.attacks.utils import get_dfs_for_victim_and_adv, get_train_config_for_adv
from distribution_inference.config import DatasetConfig, AttackConfig, BlackBoxAttackConfig, TrainConfig
from distribution_inference.utils import flash_utils,ensure_dir_exists
from distribution_inference.logging.core import AttackResult
if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "--en", help="experiment name",
        type=str, required=True)
    parser.add_argument(
        "--load_config", help="Specify config file",
        type=Path, required=True)
    parser.add_argument('--gpu',
                        default='0,1,2,3', help="device number")
    parser.add_argument(
        "--victim_path", help="path to victim'smodels directory",
        type=str, default=None)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    attack_config: AttackConfig = AttackConfig.load(
        args.load_config, drop_extra_fields=False)
    # Extract configuration information from config file
    bb_attack_config: BlackBoxAttackConfig = attack_config.black_box
    train_config: TrainConfig = attack_config.train_config
    data_config: DatasetConfig = train_config.data_config
    if train_config.misc_config is not None:
        # TODO: Figure out best place to have this logic in the module
        if train_config.misc_config.adv_config:
            # Scale epsilon by 255 if requested
            if train_config.misc_config.adv_config.scale_by_255:
                train_config.misc_config.adv_config.epsilon /= 255
    assert not train_config.save_every_epoch,"Not implemented yet"# need extra handling
    # Print out arguments
    flash_utils(attack_config)
    # Define logger
    logger = AttackResult(args.en, attack_config)

    # Get dataset wrapper
    ds_wrapper_class = get_dataset_wrapper(data_config.name)

    # Get dataset info object
    ds_info = get_dataset_information(data_config.name)()

    # Create new DS object for both and victim
    data_config_adv_1, data_config_vic_1 = get_dfs_for_victim_and_adv(
        data_config)
    ds_vic_1 = ds_wrapper_class(
        data_config_vic_1,
        skip_data=True,
        label_noise=train_config.label_noise)
    ds_adv_1 = ds_wrapper_class(data_config_adv_1)
    train_adv_config = get_train_config_for_adv(train_config, attack_config)
    attack_config.save("./a.json",indent=4)
    def single_evaluation(models_1_path=None, models_2_paths=None):
        # Load victim models for first value
        models_vic_1 = ds_vic_1.get_models(
            train_config,
            n_models=attack_config.num_victim_models,
            on_cpu=attack_config.on_cpu,
            shuffle=False,
            epochwise_version=attack_config.train_config.save_every_epoch,
            model_arch=attack_config.victim_model_arch,
            custom_models_path=models_1_path)
        preds_a = {}
        preds_v = {}
        # For each value (of property) asked to experiment with
        for prop_value in attack_config.values:
            data_config_adv_2, data_config_vic_2 = get_dfs_for_victim_and_adv(
                data_config, prop_value=prop_value)

            # Create new DS object for both and victim (for other ratio)
            ds_vic_2 = ds_wrapper_class(
                data_config_vic_2, skip_data=True,
                label_noise=train_config.label_noise)
            ds_adv_2 = ds_wrapper_class(data_config_adv_2)

            # Load victim models for other value
            models_vic_2 = ds_vic_2.get_models(
                train_config,
                n_models=attack_config.num_victim_models,
                on_cpu=attack_config.on_cpu,
                shuffle=False,
                epochwise_version=attack_config.train_config.save_every_epoch,
                model_arch=attack_config.victim_model_arch,
                custom_models_path=models_2_paths[i] if models_2_paths else None)
            preds_a[prop_value] = {}
            preds_v[prop_value] = {}
            for t in range(attack_config.tries):
                print("{}: trial {}".format(prop_value, t))
                models_adv_1 = ds_adv_1.get_models(
                    train_adv_config,
                    n_models=bb_attack_config.num_adv_models,
                    on_cpu=attack_config.on_cpu,
                    model_arch=attack_config.adv_model_arch)
                models_adv_2 = ds_adv_2.get_models(
                    train_adv_config,
                    n_models=bb_attack_config.num_adv_models,
                    on_cpu=attack_config.on_cpu,
                    model_arch=attack_config.adv_model_arch)

                # Get victim and adv predictions on loaders for first ratio
                preds_adv_on_1, preds_vic_on_1, ground_truth_1 = get_vic_adv_preds_on_distr(
                    models_vic=(models_vic_1, models_vic_2),
                    models_adv=(models_adv_1, models_adv_2),
                    ds_obj=ds_adv_1,
                    batch_size=bb_attack_config.batch_size,
                    epochwise_version=attack_config.train_config.save_every_epoch,
                    preload=bb_attack_config.preload,
                    multi_class=bb_attack_config.multi_class,
                    make_processed_version=attack_config.victim_processed_variant
                )
                # Get victim and adv predictions on loaders for second ratio
                preds_adv_on_2, preds_vic_on_2, ground_truth_2 = get_vic_adv_preds_on_distr(
                    models_vic=(models_vic_1, models_vic_2),
                    models_adv=(models_adv_1, models_adv_2),
                    ds_obj=ds_adv_2,
                    batch_size=bb_attack_config.batch_size,
                    epochwise_version=attack_config.train_config.save_every_epoch,
                    preload=bb_attack_config.preload,
                    multi_class=bb_attack_config.multi_class,
                    make_processed_version=attack_config.victim_processed_variant
                )
                # Wrap predictions to be used by the attack
                preds_adv = PredictionsOnDistributions(
                    preds_on_distr_1=preds_adv_on_1,
                    preds_on_distr_2=preds_adv_on_2
                )
                preds_vic = PredictionsOnDistributions(
                    preds_on_distr_1=preds_vic_on_1,
                    preds_on_distr_2=preds_vic_on_2
                )
                preds_a[prop_value][t]= preds_adv
                preds_v[prop_value][t]= preds_vic
        return preds_a,preds_v
    if args.victim_path:
        def joinpath(x, y): return os.path.join(
            args.victim_path, str(x), str(y))
        for i in range(1, 3+1):
            models_1_path = joinpath(data_config.value, i)
            model_2_paths = [joinpath(v, i) for v in attack_config.values]
            preds = single_evaluation(models_1_path, model_2_paths)
    else:
        preds = single_evaluation()
    preds_path = os.path.join(ds_info.base_models_dir,preds,args.en)
    ensure_dir_exists(preds_path)
    attack_config.save(os.path.join(preds_path,"config.json"),indent=4)
    with open(os.path.join(preds_path,"preds_a.p"),"wb") as f:
        pickle.dump(preds[0], f)
    with open(os.path.join(preds_path,"preds_v.p"),"wb") as f:
        pickle.dump(preds[1], f)
        