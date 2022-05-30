import argparse
import pickle
import torch
import copy
from calculate import run
from statistics import write


def main(args):
    path = '../' + args.model + '/layer ' + str(args.layer)
    all_embeddings = dict()
    for l in args.languages:
        filename = path + '/' + l + '_' + args.model + '_embedding.dat'
        with open(filename, 'rb') as fp:
            value = torch.tensor(pickle.load(fp))
            all_embeddings[l] = value
    args.all_embeddings = all_embeddings

    for target_language in languages:
        for feature_index in range(len(args.features)):
            feature = args.features[feature_index]
            if feature in list(args.INFO[target_language].keys()):
                for train_epoch in range(1, args.num_train_epochs + 1):
                    acc = run(target_language, feature_index, train_epoch, args)
                    with open(args.filename, 'a', encoding='utf-8') as fp:
                        fp.write("language:" + str(target_language) + " \t"
                                 + "feature_index:" + str(feature_index) + "\t"
                                 + "train_epoch:" + str(train_epoch) + "\t"
                                 + "acc:" + str(acc) + '\n')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='bert', choices=['bert', 'xlm-R'])
    parser.add_argument('--num_train_epochs', type=int, default=5)
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.5)
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--input_size', type=int, default=768, choices=[768, 1024])
    parser.add_argument('--hidden_dim', type=int, default=100) #CHANGE THIS! TO 100
    parser.add_argument('--train_batch_size', type=int, default=512)
    parser.add_argument('--layer', type=int, default=1)

    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.filename = 'result_' + args.model + '_' + 'layer' + str(args.layer) + '.txt'

    print(args)

    with open('features2num_WALS.dat', 'rb') as fp: #typological_features/
        features2num_WALS = pickle.load(fp)
    with open('features2num_SSWL.dat', 'rb') as fp: #typological_features/
        features2num_SSWL = pickle.load(fp)
    args.features2num = {**features2num_WALS, **features2num_SSWL}

    with open('WALS_INFO_NORDIC_dict.dat', 'rb') as fp: #typological_features/
        WALS_INFO = pickle.load(fp)
    with open('SSWL_INFO_NORDIC_dict.dat', 'rb') as fp: #typological_features/
        SSWL_INFO = pickle.load(fp)
    args.INFO = copy.deepcopy(WALS_INFO)
    for l in SSWL_INFO.keys():
        for f in SSWL_INFO[l].keys():
            args.INFO[l][f] = SSWL_INFO[l][f]

    args.features = list(features2num_WALS.keys()) + list(features2num_SSWL.keys())
    languages = list(WALS_INFO.keys())
    args.languages = languages

    main(args)
    write(args)
