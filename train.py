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
from utils.load_data import get_dataset
from utils.main_utils import print_log, output_namespace, check_dir, load_config
import warnings
warnings.filterwarnings('ignore')

import random 
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def set_seed(seed):
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 


from utils.recorder import Recorder
from utils import *


class Exp:
    def __init__(self, args, show_params=True):
        self.args = args
        self.config = args.__dict__
        self.device = self._acquire_device()
        self.total_step = 0
        self._preparation()
        if show_params:
            print_log(output_namespace(self.args))
    
    def _acquire_device(self):
        if self.args.use_gpu:
            torch.cuda.set_device(self.args.local_rank)
            device = torch.device("cuda", self.args.local_rank)
            # device = torch.device('cuda:0')
            print('Use GPU:',device)
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device
    
    def _preparation(self):
        self.path = osp.join(self.args.res_dir, self.args.ex_name)
        check_dir(self.path)

        self.checkpoints_path = osp.join(self.path, 'checkpoints')
        check_dir(self.checkpoints_path)

        sv_param = osp.join(self.path, 'model_param.json')
        with open(sv_param, 'w') as file_obj:
            json.dump(self.args.__dict__, file_obj)

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO, filename=osp.join(self.path, 'log.log'),
                            filemode='a', format='%(asctime)s - %(message)s')
        # prepare data
        self._get_data()
        # build the method
        self._build_method()

    def _build_method(self):
        steps_per_epoch = len(self.train_loader)
        self.method = MolDesign(self.args, self.device, steps_per_epoch)

    def _get_data(self):
        self.train_loader = get_dataset(root="./data/crossdocked_pocket10", 
                                        num_workers = self.args.num_workers,
                                        batch_size = self.args.batch_size,
                                        mode="train")

        self.test_loader = get_dataset(root="./data/crossdocked_pocket10",  
                                        num_workers = self.args.num_workers,
                                        batch_size = self.args.batch_size,
                                        mode="test")

    def _save(self, name=''):
        torch.save(self.method.model.state_dict(), osp.join(self.checkpoints_path, name + '.pth'))
        fw = open(osp.join(self.checkpoints_path, name + '.pkl'), 'wb')
        state = self.method.scheduler.state_dict()
        pickle.dump(state, fw)

    def _load(self, epoch):
        self.method.model.load_state_dict(torch.load(osp.join(self.checkpoints_path, str(epoch) + '.pth')))
        fw = open(osp.join(self.checkpoints_path, str(epoch) + '.pkl'), 'rb')
        state = pickle.load(fw)
        self.method.scheduler.load_state_dict(state)

    def train(self):
        recorder = Recorder(self.args.patience, verbose=True)
        for epoch in range(self.args.epoch):
            train_metric = self.method.train_one_epoch(self.train_loader)

            self._save("epoch_{}".format(epoch))
            if epoch % self.args.log_step == 0:
                valid_metric = self.method.valid_one_epoch(self.test_loader)
                
                print_log('Epoch: {}, Steps: {} | Train Loss: {:.4f}  Valid Loss: {:.4f} \n'.format(epoch + 1, len(self.train_loader), train_metric['loss'],  valid_metric['loss']))
                recorder(valid_metric['loss'], self.method.model, self.path)
                if recorder.early_stop:
                    print("Early stopping")
                    logging.info("Early stopping")
                    break
            
        best_model_path = osp.join(self.path, 'checkpoint.pth')
        self.method.model.load_state_dict(torch.load(best_model_path))


    def test(self, opt_config):
        epoch_metric = self.method.test_one_epoch(self.test_loader, opt_config)
        nni.report_intermediate_result(epoch_metric)
        return epoch_metric


if __name__ == '__main__':
    
    torch.distributed.init_process_group(backend='nccl')

    args = create_parser()
    config = args.__dict__

    tuner_params = nni.get_next_parameter()
    config.update(tuner_params) 
    default_params = load_config(osp.join('./configs', args.method + '.py' if args.config_file is None else args.config_file))
    config.update(default_params)
    config.update(tuner_params)
    print(config)

    set_seed(111)
    exp = Exp(args)
    
    print('>>>>>>>>>>>>>>>>>>>>>>>>>> training <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    print(torch.cuda.current_device())
    exp.train() 
   