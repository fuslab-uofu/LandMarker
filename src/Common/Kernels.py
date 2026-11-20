import torch


def GaussianKernel(pt1, pt2, SIGMA, deformation=False):

    if deformation:
        dx = pt1[2].squeeze()
        dy = pt1[1].squeeze()
        dz = pt1[0].squeeze()

        ret = torch.exp(- ((dx - pt2[0])**2 +
                            (dy - pt2[1])**2 +
                            (dz - pt2[2])**2)
                            /(2 * SIGMA**2))
    else:
        ret = torch.exp(- (( - pt2[0])**2 +
                           (pt1[1] - pt2[1])**2 +
                           (pt1[2] - pt2[2])**2)
                            /(2 * SIGMA**2))

    return ret


def MultiQuadRBF_Kernel(pt1, pt2, SIGMA, deformation=False):

    if deformation:
        dx = pt1[2].squeeze()
        dy = pt1[1].squeeze()
        dz = pt1[0].squeeze()

        ret = torch.sqrt(1 + SIGMA * (
            (dx - pt2[0])**2 + (dy - pt2[1])**2 + (dz - pt2[2])**2
        ))
    else:
        ret = torch.sqrt(1 + SIGMA * (
            (pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2 + (pt1[2] - pt2[2])**2
        ))

    return ret