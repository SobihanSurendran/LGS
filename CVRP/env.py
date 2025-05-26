# Adapted from https://github.com/yd-kwon/POMO
# Original Author: Yoon Dae Kwon
# License: MIT

from dataclasses import dataclass
import torch
from problem import get_random_problems, augment_xy_data_by_8_fold, augment_xy_data_by_4_fold


@dataclass
class Reset_State:
    depot_xy: torch.Tensor = None
    # shape: (batch, 1, 2)
    node_xy: torch.Tensor = None
    # shape: (batch, problem, 2)
    node_demand: torch.Tensor = None
    # shape: (batch, problem)


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor = None
    K_IDX: torch.Tensor = None
    # shape: (batch, K)
    selected_count: int = None
    load: torch.Tensor = None
    # shape: (batch, K)
    current_node: torch.Tensor = None
    # shape: (batch, K)
    ninf_mask: torch.Tensor = None
    # shape: (batch, K, problem+1)
    finished: torch.Tensor = None
    # shape: (batch, K)


class CVRPEnv:
    def __init__(self, config):

        # Const @INIT
        ####################################
        self.config = config
        self.problem_size = config.problem_size
        self.K = config.K
        self.device = config.device

        self.FLAG__use_saved_problems = False
        self.saved_depot_xy = None
        self.saved_node_xy = None
        self.saved_node_demand = None
        self.saved_index = None

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.K_IDX = None
        # IDX.shape: (batch, K)
        self.depot_node_xy = None
        # shape: (batch, problem+1, 2)
        self.depot_node_demand = None
        # shape: (batch, problem+1)

        # Dynamic-1
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, K)
        self.selected_node_list = None
        # shape: (batch, K, 0~)

        # Dynamic-2
        ####################################
        self.at_the_depot = None
        # shape: (batch, K)
        self.load = None
        # shape: (batch, K)
        self.visited_ninf_flag = None
        # shape: (batch, K, problem+1)
        self.ninf_mask = None
        # shape: (batch, K, problem+1)
        self.finished = None
        # shape: (batch, K)

        # states to return
        ####################################
        self.reset_state = Reset_State()
        self.step_state = Step_State()

    def use_saved_problems(self, filename, loaded_dict=None):
        self.FLAG__use_saved_problems = True

        if loaded_dict is not None:
            self.saved_depot_xy = loaded_dict['depot_xy']
            self.saved_node_xy = loaded_dict['node_xy']
            self.saved_node_demand = loaded_dict['node_demand']
        else:
            loaded_dict = torch.load(filename, map_location=self.device)
            self.saved_depot_xy = loaded_dict['depot_xy']
            self.saved_node_xy = loaded_dict['node_xy']
            self.saved_node_demand = loaded_dict['node_demand']

        self.saved_index = 0


    def load_problems(self, batch_size, aug_factor=1):
        self.batch_size = batch_size

        if not self.FLAG__use_saved_problems:
            depot_xy, node_xy, node_demand = get_random_problems(batch_size, self.problem_size)
        else:
            depot_xy = self.saved_depot_xy[self.saved_index:self.saved_index+batch_size]
            node_xy = self.saved_node_xy[self.saved_index:self.saved_index+batch_size]
            node_demand = self.saved_node_demand[self.saved_index:self.saved_index+batch_size]
            self.saved_index += batch_size


        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * aug_factor
                depot_xy = augment_xy_data_by_8_fold(depot_xy)
                node_xy = augment_xy_data_by_8_fold(node_xy)
                node_demand = node_demand.repeat(aug_factor, 1)
            if aug_factor == 4:
                self.batch_size = self.batch_size * aug_factor
                depot_xy = augment_xy_data_by_4_fold(depot_xy)
                node_xy = augment_xy_data_by_4_fold(node_xy)
                node_demand = node_demand.repeat(aug_factor, 1)
            else:
                raise NotImplementedError

        self.depot_node_xy = torch.cat((depot_xy, node_xy), dim=1).to(self.device)
        # shape: (batch, problem+1, 2)
        depot_demand = torch.zeros(size=(self.batch_size, 1)).to(self.device)
        # shape: (batch, 1)
        self.depot_node_demand = torch.cat((depot_demand, node_demand.to(self.device)), dim=1)
        # shape: (batch, problem+1)


        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.K)
        self.K_IDX = torch.arange(self.K)[None, :].expand(self.batch_size, self.K)

        self.reset_state.depot_xy = depot_xy
        self.reset_state.node_xy = node_xy
        self.reset_state.node_demand = node_demand

        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.K_IDX = self.K_IDX

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, K)
        self.selected_node_list = torch.zeros((self.batch_size, self.K, 0), dtype=torch.long).to(self.device)
        # shape: (batch, K, 0~)

        self.at_the_depot = torch.ones(size=(self.batch_size, self.K), dtype=torch.bool).to(self.device)
        # shape: (batch, K)
        self.load = torch.ones(size=(self.batch_size, self.K)).to(self.device)
        # shape: (batch, K)
        self.visited_ninf_flag = torch.zeros(size=(self.batch_size, self.K, self.problem_size+1)).to(self.device)
        # shape: (batch, K, problem+1)
        self.ninf_mask = torch.zeros(size=(self.batch_size, self.K, self.problem_size+1)).to(self.device)
        # shape: (batch, K, problem+1)
        self.finished = torch.zeros(size=(self.batch_size, self.K), dtype=torch.bool).to(self.device)
        # shape: (batch, K)

        reward = None
        done = False
        return self.reset_state, reward, done

    def pre_step(self):
        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished

        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected):
        # selected.shape: (batch, K)

        # Dynamic-1
        ####################################
        self.selected_count += 1
        self.current_node = selected.to(self.device)
        # shape: (batch, K)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # shape: (batch, K, 0~)

        # Dynamic-2
        ####################################
        self.at_the_depot = (selected == 0)

        demand_list = self.depot_node_demand[:, None, :].expand(self.batch_size, self.K, -1).to(self.device)
        # shape: (batch, K, problem+1)
        gathering_index = selected[:, :, None].to(self.device)
        # shape: (batch, K, 1)
        selected_demand = demand_list.gather(dim=2, index=gathering_index).squeeze(dim=2)
        # shape: (batch, K)
        
        self.load -= selected_demand
        self.load[self.at_the_depot] = 1 # refill loaded at the depot

        self.visited_ninf_flag[self.BATCH_IDX, self.K_IDX, selected] = float('-inf')
        # shape: (batch, K, problem+1)
        self.visited_ninf_flag[:, :, 0][~self.at_the_depot] = 0  # depot is considered unvisited, unless you are AT the depot

        self.ninf_mask = self.visited_ninf_flag.clone()
        round_error_epsilon = 0.00001
        demand_too_large = self.load[:, :, None] + round_error_epsilon < demand_list
        # shape: (batch, K, problem+1)
        self.ninf_mask[demand_too_large] = float('-inf')
        # shape: (batch, K, problem+1)

        newly_finished = (self.visited_ninf_flag == float('-inf')).all(dim=2)
        # shape: (batch, K)
        self.finished = self.finished + newly_finished
        # shape: (batch, K)

        # do not mask depot for finished episode.
        self.ninf_mask[:, :, 0][self.finished] = 0

        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished

        # returning values
        done = self.finished.all()
        if done:
            reward = self._get_travel_distance()  # note the minus sign!
            reward = -reward
        else:
            reward = None

        return self.step_state, reward, done

    def _get_travel_distance(self):
        gathering_index = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
        # shape: (batch, K, selected_list_length, 2)
        all_xy = self.depot_node_xy[:, None, :, :].expand(-1, self.K, -1, -1)
        # shape: (batch, K, problem+1, 2)

        ordered_seq = all_xy.gather(dim=2, index=gathering_index)
        # shape: (batch, K, selected_list_length, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq-rolled_seq)**2).sum(3).sqrt()
        # shape: (batch, K, selected_list_length)

        travel_distances = segment_lengths.sum(2)
        # shape: (batch, K)
        return travel_distances

