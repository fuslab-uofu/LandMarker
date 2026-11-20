import torch


def checkBounds(im, grid, s, dims, sample_pts):
    if sample_pts:
        if dims == 2:
            im[0][grid[0] > (s[0] - 1)] = 0
            im[1][grid[1] > (s[1] - 1)] = 0

            im[0][grid[0] < 0] = 0
            im[1][grid[1] < 0] = 0
        elif dims == 3:
            im[0][grid[0] > (s[2] - 1)] = 0
            im[1][grid[1] > (s[1] - 1)] = 0
            im[2][grid[2] > (s[0] - 1)] = 0

            im[0][grid[0] < 0] = 0
            im[1][grid[1] < 0] = 0
            im[2][grid[2] < 0] = 0

    else:
        if dims == 2:
            im[grid[0] > (s[0] - 1)] = 0
            im[grid[1] > (s[1] - 1)] = 0

            im[grid[0] < 0] = 0
            im[grid[1] < 0] = 0

        elif dims == 3:
            im[grid[0] > (s[2] - 1)] = 0
            im[grid[1] > (s[1] - 1)] = 0
            im[grid[2] > (s[0] - 1)] = 0

            im[grid[0] < 0] = 0
            im[grid[1] < 0] = 0
            im[grid[2] < 0] = 0

    return im


def interp(im, grid, sample_pts=False):
    # MMKline - added device as parameter
    device = im.device

    # Defensive: ensure `grid` (used to compute indices/weights) is on the same device
    # as `im` (the tensor being indexed). Otherwise PyTorch will throw:
    # "indices should be either on cpu or on the same device as the indexed tensor".
    if isinstance(grid, torch.Tensor) and grid.device != device:
        grid = grid.to(device)

    if sample_pts:
        s = im.shape[1:]
    else:
        s = im.shape

    dims = len(s)

    if dims == 2:
        grid = grid - 1

        # Compute the floor and ceiling indices for the grid coordinate
        floor_indices = torch.floor(grid).long()
        ceil_indices = torch.ceil(grid).long()

        # Clamp the indices to the image dimensions
        floor_indices[0] = torch.clamp(floor_indices[0], 0, s[0])
        floor_indices[1] = torch.clamp(floor_indices[1], 0, s[1])
        ceil_indices[0] = torch.clamp(ceil_indices[0], 0, s[0])
        ceil_indices[1] = torch.clamp(ceil_indices[1], 0, s[1])

        padding_dims = (0, 1, 0, 1)

        im = torch.nn.functional.pad(im, padding_dims, mode='constant', value=0)

        # Compute the weights for linear interpolation
        dx = grid[0] - floor_indices[0].float()
        dy = grid[1] - floor_indices[1].float()
        w1 = (1 - dx) * (1 - dy)
        w2 = (1 - dx) * dy
        w3 = dx * (1 - dy)
        w4 = dx * dy

        w1 = w1.to(device)
        w2 = w2.to(device)
        w3 = w3.to(device)
        w4 = w4.to(device)

        if sample_pts:
            i1 = im[:, floor_indices[0], floor_indices[1]]
            i2 = im[:, floor_indices[0], ceil_indices[1]]
            i3 = im[:, ceil_indices[0], floor_indices[1]]
            i4 = im[:, ceil_indices[0], ceil_indices[1]]

            i1 = checkBounds(i1, grid, s, dims, sample_pts)
            i2 = checkBounds(i2, grid, s, dims, sample_pts)
            i3 = checkBounds(i3, grid, s, dims, sample_pts)
            i4 = checkBounds(i4, grid, s, dims, sample_pts)

            i_ret = i1 * w1 + i2 * w2 + i3 * w3 + i4 * w4

        else:
            i1 = im[floor_indices[0], floor_indices[1]]
            i2 = im[floor_indices[0], ceil_indices[1]]
            i3 = im[ceil_indices[0], floor_indices[1]]
            i4 = im[ceil_indices[0], ceil_indices[1]]

            i1 = checkBounds(i1, grid, s, dims, sample_pts)
            i2 = checkBounds(i2, grid, s, dims, sample_pts)
            i3 = checkBounds(i3, grid, s, dims, sample_pts)
            i4 = checkBounds(i4, grid, s, dims, sample_pts)

            i_ret = i1 * w1 + i2 * w2 + i3 * w3 + i4 * w4

    elif dims == 3:
        grid = grid - 1

        # Compute the floor and ceiling indices for the grid coordinate
        floor_indices = torch.floor(grid).long()
        ceil_indices = torch.ceil(grid).long()

        # Clamp the indices to the image dimensions
        floor_indices[0] = torch.clamp(floor_indices[0], 0, s[2])
        floor_indices[1] = torch.clamp(floor_indices[1], 0, s[1])
        floor_indices[2] = torch.clamp(floor_indices[2], 0, s[0])
        ceil_indices[0] = torch.clamp(ceil_indices[0], 0, s[2])
        ceil_indices[1] = torch.clamp(ceil_indices[1], 0, s[1])
        ceil_indices[2] = torch.clamp(ceil_indices[2], 0, s[0])

        padding_dims = (0, 1, 0, 1, 0, 1)

        im = torch.nn.functional.pad(im, padding_dims, mode='constant', value=0)

        # Compute the weights for linear interpolation
        dx = grid[2] - floor_indices[2].float()
        dy = grid[1] - floor_indices[1].float()

        w1 = (1 - dx) * (1 - dy)
        w2 = (1 - dx) * dy
        w3 = dx * (1 - dy)
        w4 = dx * dy

        w1 = w1.to(device)
        w2 = w2.to(device)
        w3 = w3.to(device)
        w4 = w4.to(device)

        if sample_pts:
            i1 = im[:, floor_indices[2], floor_indices[1], floor_indices[0]]
            i2 = im[:, floor_indices[2], ceil_indices[1], floor_indices[0]]
            i3 = im[:, ceil_indices[2], floor_indices[1], floor_indices[0]]
            i4 = im[:, ceil_indices[2], ceil_indices[1], floor_indices[0]]

            i1 = checkBounds(i1, grid, s, dims, sample_pts)
            i2 = checkBounds(i2, grid, s, dims, sample_pts)
            i3 = checkBounds(i3, grid, s, dims, sample_pts)
            i4 = checkBounds(i4, grid, s, dims, sample_pts)

            c1 = i1 * w1 + i2 * w2 + i3 * w3 + i4 * w4

            i1 = im[:, floor_indices[2], floor_indices[1], ceil_indices[0]]
            i2 = im[:, floor_indices[2], ceil_indices[1], ceil_indices[0]]
            i3 = im[:, ceil_indices[2], floor_indices[1], ceil_indices[0]]
            i4 = im[:, ceil_indices[2], ceil_indices[1], ceil_indices[0]]

            i1 = checkBounds(i1, grid, s, dims, sample_pts)
            i2 = checkBounds(i2, grid, s, dims, sample_pts)
            i3 = checkBounds(i3, grid, s, dims, sample_pts)
            i4 = checkBounds(i4, grid, s, dims, sample_pts)

            c2 = i1 * w1 + i2 * w2 + i3 * w3 + i4 * w4

            dz = grid[0] - floor_indices[0].float()

            i_ret = c1 * (1 - dz) + c2 * dz
            return i_ret
        else:
            i1 = im[floor_indices[2], floor_indices[1], floor_indices[0]]
            i2 = im[floor_indices[2], ceil_indices[1], floor_indices[0]]
            i3 = im[ceil_indices[2], floor_indices[1], floor_indices[0]]
            i4 = im[ceil_indices[2], ceil_indices[1], floor_indices[0]]

            i1 = checkBounds(i1, grid, s, dims, sample_pts)
            i2 = checkBounds(i2, grid, s, dims, sample_pts)
            i3 = checkBounds(i3, grid, s, dims, sample_pts)
            i4 = checkBounds(i4, grid, s, dims, sample_pts)

            c1 = i1 * w1 + i2 * w2 + i3 * w3 + i4 * w4

            i1 = im[floor_indices[2], floor_indices[1], ceil_indices[0]]
            i2 = im[floor_indices[2], ceil_indices[1], ceil_indices[0]]
            i3 = im[ceil_indices[2], floor_indices[1], ceil_indices[0]]
            i4 = im[ceil_indices[2], ceil_indices[1], ceil_indices[0]]

            i1 = checkBounds(i1, grid, s, dims, sample_pts)
            i2 = checkBounds(i2, grid, s, dims, sample_pts)
            i3 = checkBounds(i3, grid, s, dims, sample_pts)
            i4 = checkBounds(i4, grid, s, dims, sample_pts)

            c2 = i1 * w1 + i2 * w2 + i3 * w3 + i4 * w4

            dz = grid[0] - floor_indices[0].float()

            i_ret = c1 * (1 - dz) + c2 * dz

    return i_ret


def newTransformPoints(pts, phi):
    dims = pts.shape[1]
    new_pts = torch.zeros_like(pts)

    if dims == 3:
        raise NotImplementedError
    elif dims == 2:
        for x in range(pts.shape[0]):
            new_pts[x] = interp(phi, pts[x].unsqueeze(1).float(), sample_pts=True).squeeze()

    return new_pts

