import torch
import numpy as np


def GaussianKernel(x, y, sigma, df, field=False, device='cpu'):
    dims = x.shape[0]

    if not torch.is_tensor(x):
        x = torch.from_numpy(x.reshape(-1, 1))
    if not torch.is_tensor(y):
        y = torch.from_numpy(y.reshape(-1, 1))

    if not field:
        if x.shape[-1] != 1:
            x = x.reshape(-1, 1)
        if y.shape[-1] != 1:
            y = y.reshape(-1, 1)

        if df:
            temp = 1/(np.sqrt(2 * np.pi)*sigma) * torch.exp(-sum((x-y)**2)/(2*(sigma**2)))
            p1 = ((x - y)/sigma)
            p2 = torch.matmul(p1, p1.t()) + torch.eye(dims).to(device) * ((dims - 1) - sum((x-y)**2)/(sigma**2))
            ret = temp * p2
        else:
            temp = torch.exp(-sum((x-y)**2)/(2*(sigma**2)))
            ret = torch.eye(dims).to(device) * temp

    else:
        y = y.unsqueeze(-1)
        if df:
            temp = 1 / (np.sqrt(2 * np.pi) * sigma) * torch.exp(-sum((x - y) ** 2) / (2 * (sigma ** 2)))
            p1 = ((x - y) / sigma).t().unsqueeze(1)
            e = torch.eye(dims).repeat(x.shape[1], 1, 1).to(device)
            t = ((dims - 1) - sum((x - y) ** 2) / (sigma ** 2)).unsqueeze(-1).unsqueeze(-1)
            p2 = torch.matmul(p1.transpose(1, 2), p1) + e * t
            ret = p2 * temp.unsqueeze(-1).unsqueeze(-1)
        else:
            temp = torch.exp(-sum((x-y)**2)/(2*(sigma**2)))
            e = torch.eye(dims).repeat(x.shape[1], 1, 1).to(device)
            ret = e * temp.unsqueeze(-1).unsqueeze(-1)
    return ret