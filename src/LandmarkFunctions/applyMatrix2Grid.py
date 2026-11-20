import torch


def applyMatrix2Grid(volume, affine_matrix):

    output_shape = volume.shape
    device = volume.device

    # Create a grid of coordinates
    coords = torch.meshgrid(torch.arange(output_shape[0]),
                            torch.arange(output_shape[1]),
                            torch.arange(output_shape[2]),
                            indexing='ij')
    coords = torch.stack(coords)
    coords = torch.cat((coords, torch.ones(1, *coords.shape[1:]))).to(device)

    # Apply the affine transformation to the coordinates
    transformed_coords = torch.matmul(affine_matrix, coords.reshape(4, -1))
    transformed_coords = transformed_coords[:3] / transformed_coords[3]
    transformed_coords = transformed_coords.reshape(3, *coords.shape[1:])

    transformed_coords[[0, 2], :, :, :] = transformed_coords[[2, 0], :, :, :]

    return transformed_coords
