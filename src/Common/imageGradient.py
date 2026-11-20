import torch


def imageGradient(im):
    """
        Function to take the spatial differential fot the input image/volume.
        :param im: Input image should be a torch tensor
        :return: gradient (2 x h x w) of the input image/volume (h x w).
        """
    dim = len(im.shape)

    if dim == 2:
        [gx, gy] = torch.gradient(im)
        G = torch.vstack([gx.unsqueeze(0), gy.unsqueeze(0)])

    elif dim == 3:
        [gx, gy, gz] = torch.gradient(im)
        G = torch.vstack([gz.unsqueeze(0), gy.unsqueeze(0), gx.unsqueeze(0)])

    else:
        raise NotImplementedError

    return G
