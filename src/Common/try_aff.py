import torch


def affine_transform(volume, affine_matrix, output_shape):
    # Convert the volume to a PyTorch tensor
    volume_tensor = volume.float()
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

    # Interpolate the transformed coordinates to get the transformed volume
    # transformed_volume = torch.nn.functional.grid_sample(volume_tensor, transformed_coords.permute(1, 2, 3, 0),
    #                                                      align_corners=True)
    #
    # # Convert the transformed volume back to a numpy array
    # transformed_volume = transformed_volume.squeeze().detach().numpy()

    return transformed_coords