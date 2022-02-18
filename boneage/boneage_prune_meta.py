import torch as ch
import numpy as np
import os
import argparse
from boneage_model_utils_nni import get_model_features, BASE_MODELS_DIR
from utils import PermInvModel, train_meta_model, flash_utils



if __name__ == "__main__":
    #Example run: CUDA_VISIBLE_DEVICES=1 python boneage_prune_meta.py --pruner agp --sparsity 0.95 --test-only --fine-tune-epochs 10
    parser = argparse.ArgumentParser(description='Boneage')
    parser.add_argument('--n_tries', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1200)
    parser.add_argument('--train_sample', type=int, default=800)
    parser.add_argument('--val_sample', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--start_n', type=int, default=0,
                        help="Only consider starting from this layer")
    parser.add_argument('--first_n', type=int, default=np.inf,
                        help="Only consider first N layers")
    parser.add_argument('--first', help="Ratio for D_0", default="0.5")
    parser.add_argument('--second', help="Ratio for D_1")
    parser.add_argument(
        '--prune_ratio', type=float, default=None,
        help="Prune models before training meta-models")


    # NNI's basic_pruners_torch arguments
        # dataset and model
    parser.add_argument('--dataset', type=str, default='boneage',
                        help='dataset to use, mnist, cifar10 or imagenet')
    parser.add_argument('--data-dir', type=str, default='./data/',
                        help='dataset directory')
    parser.add_argument('--model', type=str, default='bonemodel',
                        choices=['lenet', 'vgg16', 'vgg19', 'resnet18','bonemodel'],
                        help='model to use')
    parser.add_argument('--pretrained-model-dir', type=str, default=None,
                        help='path to pretrained model')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='input batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=200,
                        help='input batch size for testing')
    parser.add_argument('--experiment-data-dir', type=str, default='./experiment_data',
                        help='For saving output checkpoints')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--test-only', action='store_true', default=False,
                        help='get perf on test data')
    parser.add_argument('--multi-gpu', action='store_true', default=False,
                        help='run on mulitple gpus')
    parser.add_argument('--ratio', type=float, default=0.2,
                        help='ratio of dataset')

    # pruner
    parser.add_argument('--sparsity', type=float, default=0.5,
                        help='target overall target sparsity')
    parser.add_argument('--dependency-aware', action='store_true', default=False,
                        help='toggle dependency aware mode')
    parser.add_argument('--global-sort', action='store_true', default=False,
                        help='toggle global sort mode')
    parser.add_argument('--pruner', type=str, default='l1filter',
                        choices=['level', 'l1filter', 'l2filter', 'slim', 'agp',
                                 'fpgm', 'mean_activation', 'apoz', 'taylorfo'],
                        help='pruner to use')

    # speed-up
    parser.add_argument('--speed-up', action='store_true', default=False,
                        help='Whether to speed-up the pruned model')

    # fine-tuning
    parser.add_argument('--fine-tune-epochs', type=int, default=0,
                        help='epochs to fine tune')

    parser.add_argument('--nni', action='store_true', default=False,
                        help="whether to tune the pruners using NNi tuners")    
    args = parser.parse_args()
    flash_utils(args)


    
    for j in range(2,9): #Go through first ratios
        first = str(j/10)

        accsSecondBased = []
        for i in range(2,9): #Go through second ratios
            if (i == first): #Skip the ratio that is being tested
                continue
            
            second = str(i / 10)



            train_dir_1 = os.path.join(BASE_MODELS_DIR, "adv/%s/" % first)
            train_dir_2 = os.path.join(BASE_MODELS_DIR, "adv/%s/" % second)
            test_dir_1 = os.path.join(BASE_MODELS_DIR, "victim/%s/" % first)
            test_dir_2 = os.path.join(BASE_MODELS_DIR, "victim/%s/" % second)

            # Load models, convert to features
            dims, vecs_train_1 = get_model_features(
                train_dir_1, sparsity = args.sparsity, fine_tune_epochs = args.fine_tune_epochs,
                ratio = first, first_n=args.first_n, start_n=args.start_n, prune_ratio=args.prune_ratio,
                max_read=1000)
            _, vecs_train_2 = get_model_features(
                train_dir_2, sparsity = args.sparsity, fine_tune_epochs = args.fine_tune_epochs,
                ratio = first, first_n=args.first_n, start_n=args.start_n, prune_ratio=args.prune_ratio,
                max_read=1000)

            _, vecs_test_1 = get_model_features(
                test_dir_1, sparsity = args.sparsity,  fine_tune_epochs = args.fine_tune_epochs,
                ratio = second, first_n=args.first_n, start_n=args.start_n, prune_ratio=args.prune_ratio,
                max_read=1000)
            _, vecs_test_2 = get_model_features(
                test_dir_2, sparsity = args.sparsity, fine_tune_epochs = args.fine_tune_epochs,
                ratio = second, first_n=args.first_n, start_n=args.start_n, prune_ratio=args.prune_ratio,
                max_read=1000)

            vecs_train_1 = np.array(vecs_train_1)
            vecs_train_2 = np.array(vecs_train_2)

            Y_test = [0.] * len(vecs_test_1) + [1.] * len(vecs_test_2)
            Y_test = ch.from_numpy(np.array(Y_test)).cuda()
            X_test = vecs_test_1 + vecs_test_2
            X_test = np.array(X_test)

            print(len(X_test))
            print(len(Y_test))

            accs = []
            for i in range(args.n_tries):

                shuffled_1 = np.random.permutation(len(vecs_train_1))
                vecs_train_1_use = vecs_train_1[shuffled_1[:args.train_sample]]

                shuffled_2 = np.random.permutation(len(vecs_train_2))
                vecs_train_2_use = vecs_train_2[shuffled_2[:args.train_sample]]

                val_data = None
                if args.val_sample > 0:
                    vecs_val_1 = vecs_train_1[
                        shuffled_1[
                            args.train_sample:args.train_sample+args.val_sample]]
                    vecs_val_2 = vecs_train_2[
                        shuffled_2[
                            args.train_sample:args.train_sample+args.val_sample]]
                    X_val = np.concatenate((vecs_val_1, vecs_val_2))

                    Y_val = [0.] * len(vecs_val_1) + [1.] * len(vecs_val_2)
                    Y_val = ch.from_numpy(np.array(Y_val)).cuda()
                    val_data = (X_val, Y_val)

                # Ready train, test data
                Y_train = [0.] * len(vecs_train_1_use) + [1.] * len(vecs_train_2_use)
                Y_train = ch.from_numpy(np.array(Y_train)).cuda()
                X_train = np.concatenate((vecs_train_1_use, vecs_train_2_use))


                # Train meta-classifier model
                metamodel = PermInvModel(dims)
                metamodel = metamodel.cuda()

                # Train PIM model
                _, test_acc = train_meta_model(
                    metamodel,
                    (X_train, Y_train),
                    (X_train, Y_train), 
                    epochs=args.epochs, binary=True,
                    lr=0.001, batch_size=args.batch_size,
                    val_data=val_data,
                    eval_every=10, gpu=True)
                accs.append(test_acc)
                print("Run %d: %.2f" % (i+1, test_acc))

            accsSecondBased.append(accs) #record accuracies for this second ratio
            print(accs)

        print("First: "+ str(first))
        print("Second: " + str(second))
        print(accsSecondBased)
        import pickle
        with open("nni_results/NNI_Meta_Accuracies_" + str(args.sparsity)+ "Sparsity_" + str(first) + "first.txt", 'wb') as f:
            pickle.dump(accsSecondBased, f)