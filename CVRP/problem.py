import torch

def get_random_problems(batch_size, problem_size):
    depot_xy = torch.rand(batch_size, 1, 2)
    # shape: (batch, 1, 2)
    node_xy = torch.rand(batch_size, problem_size, 2)
    # shape: (batch, problem, 2)
    demand_scaler = {20: 30, 50: 40, 100: 50}.get(problem_size)
    if demand_scaler is None:
        raise NotImplementedError(f"Unsupported problem_size: {problem_size}")
    node_demand = torch.randint(1, 10, (batch_size, problem_size)) / demand_scaler
    # shape: (batch, problem)
    return depot_xy, node_xy, node_demand


def augment_xy_data_by_8_fold(xy_data):
    # xy_data.shape: (batch, N, 2)

    x = xy_data[:, :, [0]]
    y = xy_data[:, :, [1]]
    # x,y shape: (batch, N, 1)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    aug_xy_data = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape: (8*batch, N, 2)

    return aug_xy_data

def augment_xy_data_by_4_fold(xy_data):
    # xy_data.shape: (batch, N, 2)

    x = xy_data[:, :, [0]]
    y = xy_data[:, :, [1]]
    # x, y shape: (batch, N, 1)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((y, 1 - x), dim=2)
    dat4 = torch.cat((1 - y, 1 - x), dim=2)

    aug_xy_data = torch.cat((dat1, dat2, dat3, dat4), dim=0)
    # shape: (4*batch, N, 2)

    return aug_xy_data

