import torch

def get_random_problems(batch_size, problem_size):
    node_xy = torch.rand(size=(batch_size, problem_size, 2))
    # node_xy.shape: (batch, problem, 2)
    return node_xy


def augment_xy_data_by_8_fold(node_xy):
    # node_xy.shape: (batch, problem, 2)

    x = node_xy[:, :, [0]]
    y = node_xy[:, :, [1]]
    # x,y shape: (batch, problem, 1)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    aug_xy_data = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape: (8*batch, problem, 2)

    return aug_xy_data
