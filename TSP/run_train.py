import argparse
import logging
from train import run
from utils import *


##########################################################################################
# parameters

def parse_arguments():
    parser = argparse.ArgumentParser(description='Configuration of the training of LGS-Net for TSP')
    
    # Shared parameters
    parser.add_argument('--K', type=int, default=100, help='Number of latent samples')
    
    # Environment parameters
    parser.add_argument('--problem_size', type=int, default=100, help='Size of the problem')
    
    # Model parameters
    parser.add_argument('--embedding_dim', type=int, default=128, help='Dimension of embeddings')
    parser.add_argument('--encoder_layer_num', type=int, default=6, help='Number of encoder layers')
    parser.add_argument('--qkv_dim', type=int, default=16, help='Dimension of qkv')
    parser.add_argument('--head_num', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--logit_clipping', type=float, default=10, help='Value for logit clipping')
    parser.add_argument('--ff_hidden_dim', type=int, default=512, help='Dimension of feedforward hidden layer')
    parser.add_argument('--eval_type', type=str, default='sampling', help='Type of evaluation')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Dimension of hidden layers')
    parser.add_argument('--latent_dim', type=int, default=100, help='Dimension of latent space')

    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='Weight decay')
    parser.add_argument('--milestones', nargs='+', type=int, default=[2001,], help='Milestones for scheduler')
    parser.add_argument('--gamma', type=float, default=0.1, help='Gamma value for scheduler')
    
    # Trainer parameters
    parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs') # 8100
    parser.add_argument('--train_episodes', type=int, default=100*1000, help='Number of training episodes') # 1000 #93440
    parser.add_argument('--train_batch_size', type=int, default=100, help='Batch size for training')
    parser.add_argument('--model_load_enable', type=bool, default=False, help='Whether to load pre-trained model')
    parser.add_argument('--model_load_path', type=str, default='./result/train_tsp_n100', help='Path to pre-trained model')
    parser.add_argument('--model_load_epoch', type=int, default=2100, help='Epoch version of pre-trained model to load')

    # Loss parameters
    parser.add_argument('--beta', type=float, default=0.01, help='Entropic regularization parameter')
    parser.add_argument('--mode', choices=['mean', 'max', 'weighted_mean'], default='weighted_mean', help='Mode for computing the loss')
    
    # Logger parameters
    parser.add_argument('--desc', type=str, default='train_tsp_n100', help='Description for log file')
    parser.add_argument('--filename', type=str, default='run_log', help='Filename for log file')
    
    # Logging parameters
    parser.add_argument('--model_save_interval', type=int, default=1, help='Interval for saving the model')

    config = parser.parse_args()
    
    return config


##########################################################################################
# main

if __name__ == "__main__":

    config = parse_arguments()
    create_logger(config)
    
    logger = logging.getLogger('root')
    logger.info("Configuration:")
    for attr, value in vars(config).items():
        logger.info(f"{attr}: {value}")

    run(config)