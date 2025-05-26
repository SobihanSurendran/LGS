
import torch
import numpy as np
from logging import getLogger

from env import TSPEnv as Env
from model import TSPModel as Model
from utils import *


def run(config):
    logger = getLogger(name='trainer')
    result_folder = get_result_folder()
    result_log = LogData()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    config.device = device

    model = Model(config).to(config.device)
    env = Env(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.milestones, gamma=config.gamma)

    start_epoch = 1
    if config.model_load_enable:
        checkpoint_fullname = '{}/checkpoint-{}.pt'.format(config.model_load_path, config.model_load_epoch)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        model.load_state_dict(checkpoint.get('model_state_dict', {}), strict=False)
        start_epoch = 1 + config.model_load_epoch
        result_log.set_raw_data(checkpoint.get('result_log', {}))
        #optimizer.load_state_dict(checkpoint.get('optimizer_state_dict', {}))
        #scheduler.last_epoch = config.epochs - 1
        logger.info('Saved Model Loaded !!')

    time_estimator = TimeEstimator()

    for epoch in range(start_epoch, config.epochs+1):
        logger.info('=================================================================')
        train_score, train_loss = train_one_epoch(model, optimizer, scheduler, env, config, epoch, logger, result_log)
        result_log.append('train_score', epoch, train_score)
        result_log.append('train_loss', epoch, train_loss)

        # Logs & Checkpoint
        ############################
        elapsed_time_str, remain_time_str = time_estimator.get_est_string(epoch, config.epochs)
        logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, config.epochs, elapsed_time_str, remain_time_str))

        all_done = (epoch == config.epochs)
        model_save_interval = config.model_save_interval

        # Save Model
        if all_done or (epoch % model_save_interval) == 0:
            logger.info("Saving trained_model")
            checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'result_log': result_log.get_raw_data()
            }
            torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(result_folder, epoch))

        # All-done announcement
        if all_done:
            logger.info(" *** Training Done *** ")
            logger.info(" ** Model Saved ** ")


def train_one_epoch(model, optimizer, scheduler, env, config, epoch, logger, result_log):
    score_AM = AverageMeter()
    loss_AM = AverageMeter()

    train_num_episode = config.train_episodes
    episode = 0
    loop_cnt = 0
    while episode < train_num_episode:
        remaining = train_num_episode - episode
        batch_size = min(config.train_batch_size, remaining)

        avg_score, avg_loss = train_one_batch_RL(model, env, optimizer, config, epoch)
        score_AM.update(avg_score, batch_size)
        loss_AM.update(avg_loss, batch_size)

        episode += batch_size

        if epoch == 1:
            loop_cnt += 1
            if loop_cnt <= 10:
                logger.info('Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f},  Loss: {:.4f}'
                             .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode,
                                     score_AM.avg, loss_AM.avg))

    logger.info('Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f}'
                 .format(epoch, 100. * episode / train_num_episode,
                         score_AM.avg, loss_AM.avg))
    scheduler.step()

    return score_AM.avg, loss_AM.avg


def train_one_batch_RL(model, env, optimizer, config, epoch):
    model.train()
    env.load_problems(config.train_batch_size)

    reset_state, _, _ = env.reset()
    model.encode(reset_state)

    prob_list = torch.zeros(size=(config.train_batch_size, env.K, 0)).to(config.device)

    state, reward, done = env.pre_step()
    while not done:
        selected, prob = model(state)
        state, reward, done = env.step(selected)
        prob_list = torch.cat((prob_list, prob[:, :, None].to(config.device)), dim=2)

    advantage = reward - reward.float().mean(dim=1, keepdims=True)
    advantage /= (advantage.std(dim=1, keepdims=True) + 1e-8)

    log_prob = prob_list.log().sum(dim=2)
    log_prob = torch.clamp(prob_list.log(), min=-10, max=10).sum(dim=2)

    if config.mode == 'max':
        max_advantage_index = torch.argmax(advantage, dim=1)
        max_log_prob = log_prob[torch.arange(log_prob.size(0)), max_advantage_index]

        max_advantage = advantage[torch.arange(advantage.size(0)), max_advantage_index]
        loss = -max_advantage * max_log_prob

    if config.mode == 'mean':
        loss = -(advantage * log_prob).mean(dim=1)

    if config.mode == 'weighted_mean':
        log_weight = advantage - torch.max(advantage, 1, keepdim=True)[0]  # for stability 
        T_start = 1  #100
        decay_rate = 0.0005  #0.005
        temp = T_start * np.exp(-decay_rate / epoch)
        weight = torch.exp(log_weight / temp)
        weight = weight / torch.sum(weight, 1, keepdim=True)
        weight = weight.detach()
        loss = -((advantage * log_prob) * weight).sum(dim=1)

    entropy = -(prob_list * prob_list.log()).sum(dim=2).mean(dim=1)

    loss_mean = loss.mean() + config.beta * entropy.mean()

    max_pomo_reward, _ = reward.max(dim=1)
    score_mean = -max_pomo_reward.float().mean()

    model.zero_grad()
    loss_mean.backward()
    optimizer.step()

    return score_mean.item(), loss_mean.item()

