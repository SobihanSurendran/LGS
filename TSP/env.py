# Adapted from https://github.com/yd-kwon/POMO
# Original Author: Yoon Dae Kwon
# License: MIT

from dataclasses import dataclass
import torch
from problem import get_random_problems, augment_xy_data_by_8_fold


@dataclass
class Reset_State:
    node_xy: torch.Tensor
    # shape: (batch, problem, 2)


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor
    K_IDX: torch.Tensor
    # shape: (batch, K)
    current_node: torch.Tensor = None
    # shape: (batch, K)
    ninf_mask: torch.Tensor = None
    # shape: (batch, K, node)


class TSPEnv:
    def __init__(self, config):

        # Const @INIT
        ####################################
        self.config = config
        self.problem_size = config.problem_size
        self.K = config.K
        self.device = config.device

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.K_IDX = None
        # IDX.shape: (batch, K)
        self.FLAG__use_saved_problems = False
        self.node_xy = None
        # shape: (batch, node, node)

        # Dynamic
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, K)
        self.selected_node_list = None
        # shape: (batch, K, 0~problem)

    def use_saved_problems(self, filename):
        self.FLAG__use_saved_problems = True
        self.saved_node_xy = torch.load(filename, map_location=self.device)
        self.saved_index = 0

    def load_problems(self, batch_size, aug_factor=1):
        self.batch_size = batch_size

        if not self.FLAG__use_saved_problems:
            self.node_xy = get_random_problems(batch_size, self.problem_size).to(self.device)
            # node_xy.shape: (batch, problem, 2)
        else:
            self.node_xy = self.saved_node_xy[self.saved_index:self.saved_index + batch_size].to(self.device)
            self.saved_index += batch_size

        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                self.node_xy = augment_xy_data_by_8_fold(self.node_xy)
                # shape: (8*batch, problem, 2)
            else:
                raise NotImplementedError

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.K)
        self.K_IDX = torch.arange(self.K)[None, :].expand(self.batch_size, self.K)

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, K)
        self.selected_node_list = torch.zeros((self.batch_size, self.K, 0), dtype=torch.long).to(self.device)
        # shape: (batch, K, 0~problem)

        # CREATE STEP STATE
        self.step_state = Step_State(BATCH_IDX=self.BATCH_IDX, K_IDX=self.K_IDX)
        self.step_state.ninf_mask = torch.zeros((self.batch_size, self.K, self.problem_size)).to(self.device)
        # shape: (batch, K, problem)

        reward = None
        done = False
        return Reset_State(self.node_xy), reward, done

    def pre_step(self):
        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected):
        # selected.shape: (batch, K)

        self.selected_count += 1
        self.current_node = selected
        # shape: (batch, K)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # shape: (batch, K, 0~problem)

        # UPDATE STEP STATE
        self.step_state.current_node = self.current_node
        # shape: (batch, K)
        self.step_state.ninf_mask[self.BATCH_IDX, self.K_IDX, self.current_node] = float('-inf')
        # shape: (batch, K, node)

        # returning values
        done = (self.selected_count == self.problem_size)
        if done:
            reward = -self._get_travel_distance()  # note the minus sign!
        else:
            reward = None

        return self.step_state, reward, done

    def _get_travel_distance(self):
        gathering_index = self.selected_node_list.unsqueeze(3).expand(self.batch_size, -1, self.problem_size, 2)
        # shape: (batch, K, problem, 2)
        seq_expanded = self.node_xy[:, None, :, :].expand(self.batch_size, self.K, self.problem_size, 2)

        ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
        # shape: (batch, K, problem, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq-rolled_seq)**2).sum(3).sqrt()
        # shape: (batch, K, problem)

        travel_distances = segment_lengths.sum(2)
        # shape: (batch, K)
        return travel_distances

