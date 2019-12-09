#!/usr/bin/python3.5
import sys, timeit
from deepcloak import *
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

models=[
'models/mdl_b15openssl098_c20_s10000_f1000_h5_e100.h5',
'models/mdl_b15openssl100_c20_s10000_f1000_h5_e100.h5',
'models/mdl_b15openssl101_c20_s10000_f1000_h5_e100.h5',
'models/mdl_b15openssl102_c20_s10000_f1000_h5_e100.h5',
'models/mdl_b15openssl110_c20_s10000_f1000_h5_e100.h5'
]

data=[
'data/b15openssl098_c20_s10000_f1000_h6.h5',
'data/b15openssl100_c20_s10000_f1000_h6.h5',
'data/b15openssl101_c20_s10000_f1000_h6.h5',
'data/b15openssl102_c20_s10000_f1000_h6.h5',
'data/b15openssl110_c20_s10000_f1000_h6.h5'
]

pc = 2000

for lib in range(4,5):
	model_path = models[lib]
	data_path = data[lib]
	meta = DC_meta(data_path, 20, 10000, 1000, 5) # later try 10K, 30K samples

	craft_adversarial(model_path, meta, 'GA', distilled=0, shuffle_classes=1, pert_count=pc)
	craft_adversarial(model_path, meta, 'GSA', distilled=0, shuffle_classes=1, pert_count=pc)
	craft_adversarial(model_path, meta, 'LBFGSA', distilled=0, shuffle_classes=1, pert_count=pc)
	craft_adversarial(model_path, meta, 'SLSQPA', distilled=0, shuffle_classes=1, pert_count=pc)
	craft_adversarial(model_path, meta, 'SMA', distilled=0, shuffle_classes=1, pert_count=pc)
#	craft_adversarial(model_path, meta, 'BA', distilled=0, shuffle_classes=1, pert_count=pc)
	craft_adversarial(model_path, meta, 'GBA', distilled=0, shuffle_classes=1, pert_count=pc)
	craft_adversarial(model_path, meta, 'CRA', distilled=0, shuffle_classes=1, pert_count=pc)
	craft_adversarial(model_path, meta, 'AUNA', distilled=0, shuffle_classes=1, pert_count=pc)
	craft_adversarial(model_path, meta, 'AGNA', distilled=0, shuffle_classes=1, pert_count=pc)
	craft_adversarial(model_path, meta, 'BUNA', distilled=0, shuffle_classes=1, pert_count=pc)
	craft_adversarial(model_path, meta, 'SPNA', distilled=0, shuffle_classes=1, pert_count=pc)

