#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'Zhiquan Wang'
__maintainer__ = 'Zhiquan Wang'
__email__ = 'wang4490@purdue.edu'
__status__ = 'development'
__laboratory__ = 'hpcg lab'
__date__ = '2020/10/06-10:46 PM'

import click
import os
import numpy as np
import random

@click.group()
def cli():
    pass


@cli.command()
@click.option('--path', type=click.Path())
def relabel(path):
    label_file_path = os.path.join(path, '/org.txt')
    relabeled_file_path = os.path.join(path, '/org_relabeled.txt')
    relabeled_lines = []
    tag_map = [3, 0, 2, 0, 1, 1, 1]
    with open(label_file_path, 'r') as label_f:
        for line in label_f.readlines():
            file_name, tag = line.split(' ')
            relabeled_lines.append('{f} {t}\n'.format(f=file_name, t=tag_map[int(tag)]))
    if os.path.isfile(path=relabeled_file_path):
        os.remove(relabeled_file_path)
    with open(relabeled_file_path, 'w') as relabel_f:
        for line in relabeled_lines:
            relabel_f.write(line)


@cli.command()
@click.option('--path', type=click.Path())
def inspect(path):
    class_counter = [0] * 4
    ## Target_label = 1 -> neg-inact
    ## Target_label = 2 -> pos-inact
    ## Target_label = 3 -> pos-act
    ## Target_label = 4 -> neg-act
    #class_names = ['pleasant-active', 'pleasant-inactive', 'unpleasant-active', 'unpleasant-inactive']
    class_names = ['negative-inactive', 'positive-inactive', 'positive-active', 'negative-inactive']
    with open(path, 'r') as label_f:
        for line in label_f.readlines():
            file_name, tag = line.split(' ')
            class_counter[int(tag)] += 1
    total_num = np.sum(class_counter)
    for i in range(4):
        print('{c_name}: # {num} - {p:.2f}%'.format(c_name=class_names[i], num=str(class_counter[i]), p=class_counter[i] / total_num * 100))


@cli.command()
@click.option('--origin',type=click.Path(),help='a path of the original tag file')
@click.option('--target',type=click.Path(),help='a path of the target folder')
@click.option('--proportion','-p',type=click.FLOAT,multiple=True)
def divide(origin,target,proportion):
    origin_datasets = [[],[],[],[]]
    class_nums = [0]*4
    with open(origin, 'r') as label_f:
        for line in label_f.readlines():
            file_name, tag = line.split(' ')
            origin_datasets[int(tag)].append(file_name)
            class_nums[int(tag)] += 1
    for tags in origin_datasets:
        random.shuffle(tags)
    class_proportion = np.array(class_nums)/np.sum(class_nums)
    num_per_sets = []
    for p in proportion:
        num_per_class = []
        for i in range(4):
            num_per_class.append(int(p*class_nums[i]))
        num_per_sets.append(num_per_class)
    acum_indexes = np.cumsum(np.array(num_per_sets),axis=0)
    acum_indexes = np.vstack((np.zeros((1,4)),acum_indexes)).astype(int)
    if not os.path.isdir(target):
        os.mkdir(target)
    file_names = ['train.txt','test.txt','val.txt']
    for i in range(len(file_names)):
        file_path = os.path.join(target,file_names[i])
        with open(file_path, 'w+') as relabel_f:
            for j in range(len(acum_indexes)):
                for k in range(acum_indexes[i][j],acum_indexes[i+1][j]):
                    relabel_f.write('{f} {t}\n'.format(f=origin_datasets[j][k],t=j))

if __name__ == '__main__':
    cli()
