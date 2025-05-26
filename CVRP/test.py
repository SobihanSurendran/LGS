import dis
import math
from re import A
import time
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from env import CVRPEnv as Env
from model import CVRPModel as Model
from model import replace_decoder
from logging import getLogger
from utils import *



def run_lgs(config):
    """
    Latent Guided Sampling (LGS) for CVRP
    """
    # result folder, logger
    logger = getLogger(name='trainer') 
    result_folder = get_result_folder()

    # CUDA
    use_cuda = torch.cuda.is_available()
    config.device = torch.device("cuda:0" if use_cuda else "cpu")

    # ENV and MODEL
    env = Env(config)
    pretrained_model = Model(config).to(config.device)

    # Restore the model
    checkpoint = torch.load(config.model_load_path, map_location=config.device)
    pretrained_model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded")

    original_decoder_state_dict = pretrained_model.decoder.state_dict()

    dataset_size = config.test_episodes

    instance_costs = torch.zeros(dataset_size, device=config.device)

    if config.test_data_load_enable:
        env.use_saved_problems(config.test_data_load_filename)

    batch_size = config.test_batch_size

    total_iterations = config.mcmc_iterations + (config.mcmc_iterations+config.sa_iterations) * (config.search_iterations-1)
    if config.run_best:
        total_iterations = 303
        config.search_iterations = 7
        val_mcmc = [1, 1, 5, 15, 25, 100, 150]

    total_start_time = time.time()
    all_iteration_rewards = torch.zeros(dataset_size, total_iterations, device=config.device)

    for episode in tqdm(range(math.ceil(dataset_size / batch_size))):

        batch_size = min(batch_size, dataset_size - episode * batch_size)

        with torch.no_grad():
            #model = pretrained_model
            model = replace_decoder(pretrained_model, original_decoder_state_dict,
                                                     config).to(config.device)
            #model.eval()
            env.load_problems(batch_size, config.aug_factor)
            reset_state, _, _ = env.reset()
            model.encode(reset_state)

        max_rewards = torch.full((batch_size,), float('-inf'), device=config.device)

        batch_iteration_rewards = []
        for t in range(config.search_iterations):

            if config.run_best:
                config.mcmc_iterations = val_mcmc[t]

            # MCMC step
            ###############################################
            max_reward_mcmc, max_rewards_mcmc_per_iteration = test_batch_IMCMC(
                env, model, config, batch_size
            )

            # SA step, skip for the last iteration
            ###############################################
            if config.sa_iterations > 0 and t != config.search_iterations - 1:
                max_reward_sa, max_rewards_sa_per_iteration = SA_step(
                    env, model, config, batch_size
                )
                max_rewards = torch.maximum(max_rewards, torch.maximum(max_reward_mcmc, max_reward_sa))
                max_rewards_per_iteration = torch.cat((max_rewards_mcmc_per_iteration, max_rewards_sa_per_iteration), dim=1)
            else:
                max_rewards = torch.maximum(max_rewards, max_reward_mcmc)
                max_rewards_per_iteration = max_rewards_mcmc_per_iteration

            batch_iteration_rewards.append(max_rewards_per_iteration)

        # Store their objective function value
        instance_costs[
        episode * batch_size: episode * batch_size + batch_size] = -max_rewards
        all_iteration_rewards[episode * batch_size: episode * batch_size + batch_size] = torch.cat(batch_iteration_rewards, dim=1)


    results_dict = {
        "all_costs": all_iteration_rewards.cpu().numpy(),
        "best_costs": instance_costs.cpu().numpy(),
    }

    results_file = "results_cvrp.pkl"
    with open(results_file, "wb") as f:
        pickle.dump(results_dict, f)

    total_end_time = time.time()  
    total_duration = total_end_time - total_start_time

    logger.info(" *** Test Done *** ")
    logger.info(" NO-AUG SCORE: {:.4f} ".format(instance_costs.mean().item()))
    logger.info(f"Total time for inference: {total_duration:.4f} seconds.")



def SA_step(env, model, config, batch_size):
        batch_s = config.aug_factor * batch_size  

        optimizer = optim.Adam(
            model.decoder.multi_head_combine.parameters(), lr=config.param_lr,
            weight_decay=config.model_weight_decay)

        # Start the search
        ###############################################
        max_rewards_per_iteration = []
        for _ in range(config.sa_iterations):

            # Rollout
            ###############################################
            #with torch.no_grad():
            reset_state, _, _ = env.reset()
            state, reward, done = env.pre_step()

            prob_list = torch.zeros(size=(batch_s, env.K, 0)).to(config.device)
            while not done:
                selected, prob = model(state)
                # shape = (batch, K)
                state, reward, done = env.step(selected)
                prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

            aug_reward = reward.reshape(config.aug_factor, batch_size, -1)
            max_reward = aug_reward.max(dim=2).values.max(dim=0).values
            max_rewards_per_iteration.append(max_reward) 

            log_prob = prob_list.log().sum(dim=2)
            # shape = (batch, K)
            advantage = reward - reward.mean(dim=1, keepdim=True)

            group_loss = -advantage * log_prob
            # shape = (batch, K)
            loss = group_loss.mean()  

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        max_rewards_per_iteration = -torch.stack(max_rewards_per_iteration, dim=1)

        return max_reward, max_rewards_per_iteration


def test_batch_IMCMC(env, model, config, batch_size):

    K = config.K
    search_space_bound = config.search_space_bound
    maxiter = config.mcmc_iterations
    scaling_factor = config.proposal_scale
    proposal_crossover = config.proposal_crossover
    device = config.device
    aug_batch_size = batch_size * config.aug_factor

    # Initialize population
    population = model.z.clone().detach()
    population_cost = torch.ones((aug_batch_size, K), device=device) * 100 
    max_rewards_per_iteration = []

    for i in range(1, maxiter + 1):
        random_indices = torch.randint(0, K, (batch_size*config.aug_factor, 3, K), device=device)
        idx_0, idx_1, idx_2 = random_indices[:, 0], random_indices[:, 1], random_indices[:, 2]

        # Generates proposal candidates
        batch_idx = torch.arange(aug_batch_size).unsqueeze(-1)
        x_diff = population[batch_idx, idx_1] - population[batch_idx, idx_2]
        candidates = population[batch_idx, idx_0] + scaling_factor * x_diff
        gaussian_noise = torch.normal(0.0, 0.01, size=candidates.shape, device=device)
        candidates += gaussian_noise

        # Optional Crossover
        do_crossover = True
        if do_crossover:
            crossover_mask = torch.rand_like(candidates) > proposal_crossover  
            candidates = torch.where(crossover_mask, population, candidates)  

        candidates = torch.clamp(candidates, min=-search_space_bound, max=search_space_bound)
        model.z = candidates.clone().detach()

        # Rollout
        ###############################################
        with torch.no_grad():
            reset_state, _, _ = env.reset()

        state, reward, done = env.pre_step()

        while not done:
            selected, prob = model(state)
            # shape: (batch, K)
            state, reward, done = env.step(selected)

        scores_trial = env._get_travel_distance()

        def compute_r_t(z_t, z_t_minus_1, cost_t, cost_t_minus_1, lambda_):
            mu = model.mu.unsqueeze(1).repeat(1, config.K, 1)
            log_var = model.log_var.unsqueeze(1).repeat(1, config.K, 1)
            sigma = torch.exp(0.5 * log_var)  
            normal_dist = torch.distributions.Normal(mu, sigma)  

            # Compute PDF for z_t and z_{t-1}
            pdf_t = normal_dist.log_prob(z_t).sum(dim=-1)  
            pdf_t_minus_1 = normal_dist.log_prob(z_t_minus_1).sum(dim=-1) 
            exp_term_1 = torch.exp(0.00001*(pdf_t - pdf_t_minus_1))

            # Compute the exponential cost term
            exp_term_2 = torch.exp(-(cost_t - cost_t_minus_1)/lambda_)
            r_t = exp_term_1 * exp_term_2

            p_t = torch.minimum(torch.ones_like(r_t), r_t)

            return p_t

        temp = [1e-4 * (0.9**t) for t in range(maxiter)]
        p_t = compute_r_t(candidates, population, scores_trial, population_cost, lambda_=temp[i-1])
        p_t = torch.nan_to_num(p_t, nan=0.0)

        selection_mask = torch.bernoulli(p_t).unsqueeze(-1)
        population = population * (1 - selection_mask) + candidates * selection_mask
        population_cost = population_cost * (1 - selection_mask.squeeze(-1)) + scores_trial * selection_mask.squeeze(-1)

        max_rewards_per_iteration.append(scores_trial.min(dim=1).values)

    #model.z = population.clone().detach()
            
    max_rewards_per_iteration = torch.stack(max_rewards_per_iteration, dim=1)
    max_rewards_per_iteration = max_rewards_per_iteration.view(config.aug_factor, batch_size, -1)
    max_rewards_per_iteration = max_rewards_per_iteration.min(dim=0).values

    aug_reward = -population_cost.view(config.aug_factor, batch_size, K)
    max_reward = aug_reward.max(dim=2).values.max(dim=0).values

    return max_reward, max_rewards_per_iteration
