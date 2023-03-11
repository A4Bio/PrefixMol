import argparse

from matplotlib.pyplot import plasma


def create_parser():
    parser = argparse.ArgumentParser()
    # Set-up parameters
    parser.add_argument('--device', default='cuda', type=str, help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument("--local_rank", default=-1, type=int, help="Used for DDP, local rank means the process number of each machine")
    parser.add_argument('--display_step', default=10, type=int, help='Interval in batches between display of training metrics')
    parser.add_argument('--res_dir', default='./results', type=str)
    parser.add_argument('--ex_name', default='pocket2smiles_verify_metrics', type=str)
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--gpu', default=8, type=int)
    parser.add_argument('--seed', default=111, type=int)
    # CATH

    # dataset parameters
    parser.add_argument('--data_root', default='./data/') 
    parser.add_argument('--batch_size', default=128, type=int)# 不报CUDA out of memory就往上怼
    parser.add_argument('--num_workers', default=32, type=int)
    parser.add_argument('--given_rc', default=0, type=int)
    parser.add_argument('--use_motif_action', default=0, type=int)
    parser.add_argument('--use_motif_feature', default=0, type=int)
    parser.add_argument('--use_hierachical_action', default=0, type=int)
    
    
    
    # method parameters
    parser.add_argument('--method', default='MolDesign', choices=["MolDesign"])
    parser.add_argument('--config_file', '-c', default=None, type=str)
    parser.add_argument('--hidden_dim', default=1024, type=int)
    parser.add_argument('--sparse', default=1, type=int)


    # Training parameters
    parser.add_argument('--epoch', default=50, type=int, help='end epoch')#we can have a try with 300 -2022/12/19 task
    parser.add_argument('--log_step', default=1, type=int)
    parser.add_argument('--lr', default=0.00001, type=float, help='Learning rate')#try 0.001, 0.00001, 0.0002
    parser.add_argument('--patience', default=100, type=int)
    

    args = parser.parse_args()
    return args