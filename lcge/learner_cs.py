# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
from typing import Dict
import logging
from numpy.core.fromnumeric import _size_dispatcher
import torch
from torch import optim

from datasets_lcge import TemporalDataset
from optimizers_cs import TKBCOptimizer, IKBCOptimizer
from models_cs import LCGE
from regularizers import N3, Lambda3
import os

import sys

import json


def save_best_model(model, params, best_mrr, best_hit, dataset_name, model_name):
    output_dir = "best_model"
    os.makedirs(output_dir, exist_ok=True)

    file_name = "best_model_{}_{}.pt".format(model_name, dataset_name)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "params": params,
            "best_mrr": best_mrr,
            "best_hit": best_hit,
        },
        os.path.join(output_dir, file_name),
    )
    print(f"The best model and parameters have been saved to {file_name}.")


def main(args):
    weight_static_values = [0.0, 0.3, 0.6, 0.9, 1.2, 1.5]

    best_global_mrr = 0.0
    best_global_hit = 0.0
    best_global_params = None
    best_global_model_state = None
    for weight_static in weight_static_values:

        print(f"Training with weight_static: {weight_static}")

        args.weight_static = weight_static

        dataset = TemporalDataset(args.dataset)
        sizes = dataset.get_shape()
        print("sizes of dataset is:\t", sizes)

        model = {
            'LCGE': LCGE(sizes, args.rank, args.weight_static, no_time_emb=args.no_time_emb),
        }[args.model]
        model = model.cuda()

        opt = optim.Adagrad(model.parameters(), lr=args.learning_rate)

        emb_reg = N3(args.emb_reg)
        time_reg = Lambda3(args.time_reg)

        best_mrr = 0.
        best_hit = 0.
        early_stopping = 0

        for epoch in range(args.max_epochs):
            examples = torch.from_numpy(
                dataset.get_train().astype('int64')
            )
            #print("\nexamples:\n", examples.size())

            model.train()
            if dataset.has_intervals():
                optimizer = IKBCOptimizer(
                    model, emb_reg, time_reg, opt, dataset,
                    batch_size=args.batch_size
                )
                optimizer.epoch(examples)

            else:
                optimizer = TKBCOptimizer(
                    model, emb_reg, time_reg, opt,
                    batch_size=args.batch_size
                )
                optimizer.epoch(examples)


            def avg_both(mrrs: Dict[str, float], hits: Dict[str, torch.FloatTensor]):
                """
                aggregate metrics for missing lhs and rhs
                :param mrrs: d
                :param hits:
                :return:
                """
                m = (mrrs['lhs'] + mrrs['rhs']) / 2.
                h = (hits['lhs'] + hits['rhs']) / 2.
                return {'MRR_all': m, 'hits@_all': h}

            if epoch < 0 or (epoch + 1) % args.valid_freq == 0:
                if dataset.has_intervals():
                    valid, test, train = [
                        dataset.eval(model, split, -1 if split != 'train' else 50000)
                        for split in ['valid', 'test', 'train']
                    ]
                    print('-----------------------------------------')
                    print("valid: ", valid)
                    print("test: ", test)
                    print("train: ", train)
                    print('*****************************************')

                    if test['MRR_all'] > best_mrr:
                        best_mrr = test['MRR_all']
                        best_hit = test['hits@_all']
                        early_stopping = 0
                    else:
                        early_stopping += 1
                    if early_stopping > 10:
                        print("early stopping!")
                        break

                else:
                    valid, test, train = [
                        avg_both(*dataset.eval(model, split, -1 if split != 'train' else 50000))
                        for split in ['valid', 'test', 'train']
                    ]
                    print("epoch: ", epoch+1)
                    print("valid: ", valid['MRR_all'])
                    print("test: ", test['MRR_all'])
                    print("train: ", train['MRR_all'])

                    print("test hits@n:\t", test['hits@_all'])

        if best_mrr > best_global_mrr:
            best_global_mrr = best_mrr
            best_global_hit = best_hit
            best_global_params = {'weight_static': weight_static}
            save_best_model(model, best_global_params, best_global_mrr, best_global_hit, args.dataset, args.model)

            print("The best test mrr for weight static:{} is:\t".format(weight_static), best_mrr)
            print("The best test hits@1,3,10 for weight static:{} are:\t".format(weight_static), best_hit)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Commonsense-Guided Temporal KGE"
    )
    parser.add_argument(
        '--dataset', default='wikidata12k', type=str,
        help="Dataset name"
    )
    models = [
        'LCGE'
    ]
    parser.add_argument(
        '--model', choices=models,
        help="Model in {}".format(models)
    )
    parser.add_argument(
        '--max_epochs', default=1, type=int,
        help="Number of epochs."
    )
    parser.add_argument(
        '--valid_freq', default=1, type=int,
        help="Number of epochs between each valid."
    )
    parser.add_argument(
        '--rank', default=100, type=int,
        help="Factorization rank."
    )
    parser.add_argument(
        '--batch_size', default=1000, type=int,
        help="Batch size."
    )
    parser.add_argument(
        '--learning_rate', default=1e-1, type=float,
        help="Learning rate"
    )
    parser.add_argument(
        '--emb_reg', default=0., type=float,
        help="Embedding regularizer strength"
    )
    parser.add_argument(
        '--time_reg', default=0., type=float,
        help="Timestamp regularizer strength"
    )
    parser.add_argument(
        '--no_time_emb', default=False, action="store_true",
        help="Use a specific embedding for non temporal relations"
    )
    parser.add_argument(
        '--rule_reg', default=0., type=float,
        help="Rule regularizer strength"
    )
    parser.add_argument(
        '--weight_static', default=0., type=float,
        help="Weight of static score"
    )

    args, unknown = parser.parse_known_args()

    main(args)