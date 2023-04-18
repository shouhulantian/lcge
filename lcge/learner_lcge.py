# Copyright (c) Facebook, Inc. and its affiliates.

import os
import argparse
from typing import Dict
import logging
from numpy.core.fromnumeric import _size_dispatcher
import torch
from torch import optim

from datasets_lcge import TemporalDataset
from optimizers_lcge import TKBCOptimizer, IKBCOptimizer
from models_lcge import LCGE
from regularizers_rule import N3, Lambda3

import sys

from regularizers_rule import RuleSim
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
    weight_static_values = [0.9]

    best_global_mrr = 0.0
    best_global_hit = 0.0
    best_global_params = None
    best_global_model_state = None
    for weight_static in weight_static_values:

        print(f"Training with weight_static: {weight_static}")

        args.weight_static = weight_static

        # traning begin
        dataset = TemporalDataset(args.dataset)

        with open("./src_data/rulelearning/" + args.dataset + "/rule1_p1.json", 'r') as load_rule1_p1:
            rule1_p1 = json.load(load_rule1_p1)
        with open("./src_data/rulelearning/" + args.dataset + "/rule1_p2.json", 'r') as load_rule1_p2:
            rule1_p2 = json.load(load_rule1_p2)

        f = open("./src_data/rulelearning/" + args.dataset + "/rule2_p1.txt", 'r')
        rule2_p1 = {}
        for line in f:
            head, body1, body2, confi = line.strip().split("\t")
            head, body1, body2, confi = int(head), int(body1), int(body2), float(confi)
            if head not in rule2_p1:
                rule2_p1[head] = {}
            rule2_p1[head][(body1, body2)] = confi
        f.close()

        f = open("./src_data/rulelearning/" + args.dataset + "/rule2_p2.txt", 'r')
        rule2_p2 = {}
        for line in f:
            head, body1, body2, confi = line.strip().split("\t")
            head, body1, body2, confi = int(head), int(body1), int(body2), float(confi)
            if head not in rule2_p2:
                rule2_p2[head] = {}
            rule2_p2[head][(body1, body2)] = confi
        f.close()

        f = open("./src_data/rulelearning/" + args.dataset + "/rule2_p3.txt", 'r')
        rule2_p3 = {}
        for line in f:
            head, body1, body2, confi = line.strip().split("\t")
            head, body1, body2, confi = int(head), int(body1), int(body2), float(confi)
            if head not in rule2_p3:
                rule2_p3[head] = {}
            rule2_p3[head][(body1, body2)] = confi
        f.close()

        f = open("./src_data/rulelearning/" + args.dataset + "/rule2_p4.txt", 'r')
        rule2_p4 = {}
        for line in f:
            head, body1, body2, confi = line.strip().split("\t")
            head, body1, body2, confi = int(head), int(body1), int(body2), float(confi)
            if head not in rule2_p4:
                rule2_p4[head] = {}
            rule2_p4[head][(body1, body2)] = confi
        f.close()

        rules = (rule1_p1, rule1_p2, rule2_p1, rule2_p2, rule2_p3, rule2_p4)

        sizes = dataset.get_shape()
        print("sizes of dataset is:\t", sizes)
        model = {
            'LCGE': LCGE(sizes, args.rank, rules, args.weight_static, no_time_emb=args.no_time_emb),
        }[args.model]
        model = model.cuda()

        opt = optim.Adagrad(model.parameters(), lr=args.learning_rate)

        emb_reg = N3(args.emb_reg)
        time_reg = Lambda3(args.time_reg)
        rule_reg = RuleSim(args.rule_reg)  # relation embedding reglu via rules

        best_mrr = 0.
        best_hit = 0.
        early_stopping = 0

        for epoch in range(args.max_epochs):
            examples = torch.from_numpy(
                dataset.get_train().astype('int64')
            )
            # print("\nexamples:\n", examples.size())

            model.train()
            if dataset.has_intervals():
                optimizer = IKBCOptimizer(
                    model, emb_reg, time_reg, opt, dataset,
                    batch_size=args.batch_size
                )
                optimizer.epoch(examples)

            else:
                optimizer = TKBCOptimizer(
                    model, emb_reg, time_reg, rule_reg, opt,
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
                return {'MRR': m, 'hits@[1,3,10]': h}

            if epoch < 0 or (epoch + 1) % args.valid_freq == 0:
                if dataset.has_intervals():
                    valid, test, train = [
                        dataset.eval(model, split, -1 if split != 'train' else 50000)
                        for split in ['valid', 'test', 'train']
                    ]
                    print("valid: ", valid)
                    print("test: ", test)
                    print("train: ", train)

                else:
                    valid, test, train = [
                        avg_both(*dataset.eval(model, split, -1 if split != 'train' else 50000))
                        for split in ['valid', 'test', 'train']
                    ]
                    print("epoch: ", epoch + 1)
                    print("valid: ", valid['MRR'])
                    print("test: ", test['MRR'])
                    print("train: ", train['MRR'])

                    print("test hits@n:\t", test['hits@[1,3,10]'])
                    if test['MRR'] > best_mrr:
                        best_mrr = test['MRR']
                        best_hit = test['hits@[1,3,10]']
                        early_stopping = 0
                    else:
                        early_stopping += 1
                    if early_stopping > 10:
                        print("early stopping!")
                        break

        if best_mrr > best_global_mrr:
            best_global_mrr = best_mrr
            best_global_hit = best_hit
            best_global_params = {'weight_static': weight_static}
            #save_best_model(model, best_global_params, best_global_mrr, best_global_hit, args.dataset, args.model)

            print("The best test mrr for weight static:{} is:\t".format(weight_static), best_mrr)
            print("The best test hits@1,3,10 for weight static:{} are:\t".format(weight_static), best_hit)

    print(best_global_mrr)
    print(best_global_hit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Logic and Commonsense-Guided Temporal KGE"
    )
    parser.add_argument(
        '--dataset', default='ICEWS14', type=str,
        help="Dataset name"
    )
    models = [
        'LCGE'
    ]
    parser.add_argument(
        '--model', default='LCGE', choices=models,
        help="Model in {}".format(models)
    )
    parser.add_argument(
        '--max_epochs', default=50, type=int,
        help="Number of epochs."
    )
    parser.add_argument(
        '--valid_freq', default=5, type=int,
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
        '--weight_static', type=float,
        help="Weight of static score"
    )

    args, unknown = parser.parse_known_args()

    main(args)


# load best model
"""
checkpoint = torch.load("best_model/best_model.pt")
model.load_state_dict(checkpoint["model_state_dict"])
best_params = checkpoint["params"]
best_mrr = checkpoint["best_mrr"]
best_hit = checkpoint["best_hit"]

print("Loaded best model with parameters:")
print(best_params)
print(f"Best MRR: {best_mrr}")
print(f"Best Hit: {best_hit}")

"""

