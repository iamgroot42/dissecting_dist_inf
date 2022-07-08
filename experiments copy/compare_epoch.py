from simple_parsing import ArgumentParser
from pathlib import Path
import os
import numpy as np
from distribution_inference.datasets.utils import get_dataset_wrapper, get_dataset_information
from distribution_inference.attacks.utils import get_dfs_for_victim_and_adv, get_train_config_for_adv
from distribution_inference.config import DatasetConfig, TrainConfig,MiscTrainConfig
from distribution_inference.utils import flash_utils
from distribution_inference.attacks.blackbox.per_point import np_compute_losses
from distribution_inference.attacks.blackbox.utils import get_preds_epoch_on_dis
from distribution_inference.attacks.blackbox.core import PredictionsOnDistributions
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200
def get_df(preds,gt,r1,r2,columns):
    df1 = []
    df2 = []
    es = len(preds)
    r = (r1,r2)
    for i in range(es):
       
        p = preds[i]
        #pij: ith distri of data, jth distri of model
        p11 = (p.preds_on_distr_1.preds_property_1)
        p12 = (p.preds_on_distr_1.preds_property_2)
        p21 = (p.preds_on_distr_2.preds_property_1)
        p22 = (p.preds_on_distr_2.preds_property_2)
        p11 = np_compute_losses(p11,gt[0],multi_class=False)
        p12 = np_compute_losses(p11,gt[0],multi_class=False)
        p21 = np_compute_losses(p11,gt[1],multi_class=False)
        p22 = np_compute_losses(p11,gt[1],multi_class=False)
        
        p11 = np.transpose(p.preds_on_distr_1.preds_property_1)
        p12 = np.transpose(p.preds_on_distr_1.preds_property_2)
        p21 = np.transpose(p.preds_on_distr_2.preds_property_1)
        p22 = np.transpose(p.preds_on_distr_2.preds_property_2)
        
        ps = [p11,p21,p12,p22]
        ps = [np.average(x,axis=0) for x in ps]
        for j in range(len(r)):
            for k in range(len(ps[0])):
                df1.append({
                columns[0]: str(i),
                columns[1]: ps[0+j][k],
                columns[2]: float(r[j])
                })
                df2.append({
                columns[0]: str(i),
                columns[1]: ps[2+j][k],
                columns[2]: float(r[j])
                })
    return (pd.DataFrame(df1), pd.DataFrame(df2))

if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "--load_config", help="Specify config file",
        type=Path, required=True)
    parser.add_argument('--gpu',
                        default='0,1,2,3', help="device number")
    parser.add_argument("--ratio",
    default=0.0,type=float)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    train_config = TrainConfig.load(args.load_config, drop_extra_fields=False)

    # Extract configuration information from config file
    dp_config = None
    train_config: TrainConfig = train_config
    data_config: DatasetConfig = train_config.data_config
    misc_config: MiscTrainConfig = train_config.misc_config
    # Print out arguments
    flash_utils(train_config)
    ds_wrapper_class = get_dataset_wrapper(data_config.name)

    # Get dataset info object
    ds_info = get_dataset_information(
        data_config.name)(epoch_wise=True)
    data_config_adv_1, data_config_vic_1 = get_dfs_for_victim_and_adv(
        data_config)
    # Create new DS object
    ds_vic_1 = ds_wrapper_class(data_config_vic_1, skip_data=True,label_noise=train_config.label_noise,epoch=True)
    ds_adv_1 = ds_wrapper_class(data_config_adv_1,epoch=True)
    
    models_vic_1 = ds_vic_1.get_models(
            train_config,
            n_models=10,
            on_cpu=False,
            shuffle=True,
            epochwise_version=True
            )
    
    data_config_adv_2, data_config_vic_2= get_dfs_for_victim_and_adv(
            data_config, prop_value=args.ratio)
    ds_vic_2 = ds_wrapper_class(data_config_vic_2, skip_data=True,label_noise=train_config.label_noise,epoch=True)
    ds_adv_2 = ds_wrapper_class(data_config_adv_2,epoch=True)
    models_vic_2 = ds_vic_2.get_models(
            train_config,
            n_models=10,
            on_cpu=False,
            shuffle=True,
            epochwise_version=True
            )
    models_adv_1 = ds_adv_1.get_models(
            train_config,
            n_models=10,
            on_cpu=False,
            shuffle=True,
            epochwise_version=True
            )
    models_adv_2 = ds_adv_2.get_models(
            train_config,
            n_models=10,
            on_cpu=False,
            shuffle=True,
            epochwise_version=True
            )
    ds_adv_1.ratio = 1.0
    ds_adv_2.ratio = 0.0
    _, loader1 = ds_adv_1.get_loaders(batch_size=30000)
    _, loader2 = ds_adv_2.get_loaders(batch_size=30000)
    preds_vic1, ground_truth_1 = get_preds_epoch_on_dis([models_vic_1,models_vic_2],
            loader=loader1,preload=True,
                multi_class=False)
    preds_vic2, ground_truth_2 =  get_preds_epoch_on_dis([models_vic_1,models_vic_2],
            loader=loader2,preload=True,
                multi_class=False)
    preds_adv1,g1 = get_preds_epoch_on_dis([models_adv_1,models_adv_2],
            loader=loader1,preload=True,
                multi_class=False)
    preds_adv2,g2 = get_preds_epoch_on_dis([models_adv_1,models_adv_2],
            loader=loader2,preload=True,
                multi_class=False)
    assert np.array_equal(ground_truth_1,g1)
    assert np.array_equal(ground_truth_2,g2)
    preds_v = [PredictionsOnDistributions(
                preds_on_distr_1=e1,
                preds_on_distr_2=e2
            ) for e1,e2 in zip(preds_vic1,preds_vic2)]
    preds_a = [PredictionsOnDistributions(
                preds_on_distr_1=e1,
                preds_on_distr_2=e2
            ) for e1,e2 in zip(preds_adv1,preds_adv2)]
    assert len(preds_v)==len(preds_a)
    sns.set()
    figure, axes = plt.subplots(2, 2, sharex=True, figsize=(16,16))
    figure.suptitle('Comparison epochwise')
    axes[0,0].set_title('Victim on ratio {}'.format(0.5))
    axes[0,1].set_title('Victim on ratio {}'.format(args.ratio))
    axes[1,0].set_title('Adv on ratio {}'.format(0.5))
    axes[1,1].set_title('Adv on ratio {}'.format(args.ratio))
    columns = ["Epoch","Value","Ratio of input"]
    vdf = get_df(preds_v,(ground_truth_1,ground_truth_2),1.0,0.0,columns)
    adf = get_df(preds_v,(ground_truth_1,ground_truth_2),1.0,0.0,columns)
    sns.lineplot(ax=axes[0,0],data=vdf[0], x=columns[0], y=columns[1],hue=columns[2])
    sns.lineplot(ax=axes[0,1],data=vdf[1], x=columns[0], y=columns[1],hue=columns[2])
    sns.lineplot(ax=axes[1,0],data=adf[0], x=columns[0], y=columns[1],hue=columns[2])
    sns.lineplot(ax=axes[1,1],data=adf[1], x=columns[0], y=columns[1],hue=columns[2])
    figure.savefig("./epoch_analysis")

