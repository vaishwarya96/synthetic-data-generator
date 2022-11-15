#!/usr/bin/env python
# coding: utf-8


import os
import argparse
import yaml



# yaml for var
parser = argparse.ArgumentParser()
parser.add_argument('--var', required=True, type=str, help='variables')
opt, unknown = parser.parse_known_args()


DATASET_NAME = "dataset01/"

with open(opt.var) as f:
    optyaml = yaml.safe_load(f)
TAXON_FILTER_PATH = optyaml["TAXON_FILTER_PATH"]
DATASET_PATH = optyaml["DATASET_PATH"]
DATASET_DUST_PATH = [optyaml["DATASET_DUST_PATH"]]
IMAGE_REF = optyaml["IMAGE_REF"]
SAVE_PATH = optyaml["SAVE_PATH"]
TOTAL_N_IMAGES = optyaml["TOTAL_N_IMAGES"]
PERCENTAGE_VAL = optyaml["PERCENTAGE_VAL"]

print("optyaml", optyaml)


