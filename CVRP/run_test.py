import argparse
import logging
from utils import *
from test import run_lgs


##########################################################################################
# parameters

def parse_arguments():
    parser = argparse.ArgumentParser(description='Configuration of the LGS inference method for CVRP')
    
    # Shared parameters
    parser.add_argument('--K', type=int, default=600, help='Number of latent samples')
    
    # Environment parameters
    parser.add_argument('--problem_size', type=int, default=100, help='Size of the problem')
    
    # Model parameters
    parser.add_argument('--embedding_dim', type=int, default=128, help='Dimension of embeddings')
    parser.add_argument('--encoder_layer_num', type=int, default=6, help='Number of encoder layers')
    parser.add_argument('--qkv_dim', type=int, default=16, help='Dimension of qkv')
    parser.add_argument('--head_num', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--logit_clipping', type=float, default=10, help='Value for logit clipping')
    parser.add_argument('--ff_hidden_dim', type=int, default=512, help='Dimension of feedforward hidden layer')
    parser.add_argument('--eval_type', type=str, default='argmax', help='Type of evaluation')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Dimension of hidden layers')
    parser.add_argument('--latent_dim', type=int, default=100, help='Dimension of latent space')

    # SA parameters
    parser.add_argument('--model_weight_decay', type=float, default=2.466e-05, help='Weight decay for the model')
    parser.add_argument('--param_lr', type=float, default=3.714e-06, help='Learning rate for parameters')
    
    # Trainer parameters
    parser.add_argument('--test_episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--test_batch_size', type=int, default=100, help='Batch size for training')
    parser.add_argument('--model_load_path', type=str, default='./model/CVRP_100_LGS_Net.pt', help='Path to pre-trained model')
    parser.add_argument('--augmentation_enable', type=bool, default=False, help='Whether to enable data augmentation')
    parser.add_argument('--aug_factor', type=int, default=1, help='Augmentation factor')
    parser.add_argument('--test_data_load_enable', type=bool, default=True, help='Whether to load testing data')

    # Search parameters
    parser.add_argument('--mcmc_iterations', type=int, default=75, help='Number of Interacting MCMC iterations')
    parser.add_argument('--sa_iterations', type=int, default=1, help='Number of SA steps')
    parser.add_argument('--search_iterations', type=int, default=4, help='Number of search iterations')
    parser.add_argument('--run_best', type=bool, default=False, help='Whether to run the best SA update configuration')
    parser.add_argument('--search_space_bound', default=40, type=int, help='Bound for the search space')
    # Proposal parameters
    parser.add_argument('--proposal_scale', default=0.379, type=float, help='Scaling factor in the Gaussian proposal distribution')
    parser.add_argument('--proposal_crossover', default=0.997, type=float, help='Recombination factor for crossing over')

    # Logger parameters
    parser.add_argument('--filename', type=str, default='log.txt', help='Filename for log file')
    

    config = parser.parse_args()

    # Dynamically update arguments based on problem_size
    config.test_data_load_filename = f'./data/vrp{config.problem_size}_test_seed1235.pt'
    config.desc = f'test_cvrp{config.problem_size}'
    
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

    run_lgs(config)
