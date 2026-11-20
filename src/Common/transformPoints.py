import torch
from Common.Interp import *


def coords2idx(field, point, s):
    space = field.spacing

    dev = point.device

    a = field.origin
    b = field.origin + (space * (torch.tensor(s).to(dev) - 1))
    mins = torch.min(torch.vstack([a, b]), axis=0)[0]

    new_pts = (point - mins)/torch.abs(space)
    return new_pts


def transformPoints(pts, phi):
    new_pts = torch.zeros_like(pts)
    for x in range(pts.shape[0]):
        new_pts[x] = interp(phi, pts[x].flip(-1).unsqueeze(1).float(), sample_pts=True).squeeze().flip(-1)

    return new_pts


def idx2coords(idx, field, s):
    space = field.spacing

    dev = idx.device

    a = field.origin
    b = field.origin + (space * (torch.tensor(s).to(dev) - 1))
    mins = torch.min(torch.vstack([a, b]), axis=0)[0]

    ret = idx * torch.abs(space) + mins

    return ret
