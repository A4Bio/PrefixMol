import os
import nni
import logging
import pickle
import json
import torch
import torch.utils.data
import os.path as osp
from parser_1 import create_parser
from methods.MolDesign import MolDesign
from utils.scoring_func import SimilarityWithTrain
from utils.load_data import get_dataset
from utils.main_utils import print_log, output_namespace, check_dir, load_config
import warnings
warnings.filterwarnings('ignore')

import random 
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from train import set_seed, Exp

if __name__ == '__main__':
    torch.distributed.init_process_group(backend='nccl')

    args = create_parser()
    config = args.__dict__
    default_params = load_config(osp.join('./configs', args.method + '.py' if args.config_file is None else args.config_file))
    config.update(default_params)
    print(config)

    set_seed(111)
    exp = Exp(args)
    model_path = "/huyuqi/MolDesign/results/pocket2smiles_verify_metrics/checkpoint.pth"
    exp.method.model.load_state_dict(torch.load(model_path, map_location=next(exp.method.model.parameters()).device))
    
    configs = {}
    for i in range(-4,20):
        val = i*0.5
        config = {f"three+{val}": {"vina": 0, "qed":val, "sa":val, "lipinski": val, "logp":0}}
        configs.update(config)

    for ex_name, opt_config in configs.items():
    # ex_name = 5
        print(f'>>>>>>>>>>>>>>>>>>>>>>>>>> start {ex_name}  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        test_metric = exp.test(opt_config)

        with open(f"/huyuqi/MolDesign/results/verify_multi/multi_three/{ex_name}.json", "w") as f:
            json.dump(test_metric, f, indent=4)

        print(f'>>>>>>>>>>>>>>>>>>>>>>>>>> finish {ex_name}  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

