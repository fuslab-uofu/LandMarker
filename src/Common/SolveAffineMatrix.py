import numpy as np


def SolveRigidAffine(src_points, tar_points):
    '''Takes correspondance points of a source volume and a target volume
       and returns the rigid only affine matrix formatted for use with PyCA
       apply affine functions.

       src_points = points chosen in the source (moving) volume
       tar_points = points chosen in the target (fixed) volume
       type = both sets of points are assumed to be N x Dim column numpy array

       This code is an implementation of the following technical notes:
       Sorkine-Hornung, Olga, and Michael Rabinovich. "Least-squares rigid motion using svd." Computing 1 (2017): 1.'''

    if 'torch' in str(src_points.dtype):
        src_points = src_points.cpu().numpy()
        tar_points = tar_points.cpu().numpy()

    # Calculate the mean of each set of points
    src_mean = np.mean(src_points, 0)
    tar_mean = np.mean(tar_points, 0)

    # Subtract the mean so the points are centered at [0, 0, 0]
    src_zero = src_points - src_mean
    tar_zero = tar_points - tar_mean

    # Calculate the covariance matrix
    S = np.matmul(np.transpose(src_zero), tar_zero)

    # SVD of the covariance matrix
    [U, _, V] = np.linalg.svd(S)

    # Create the weights matrix and incorporate the determinant of the rotation matrix
    if np.shape(src_points)[1] == 2:
        W = np.eye(2)
        W[1, 1] = np.linalg.det(np.matmul(np.transpose(V), np.transpose(U)))

    else:
        W = np.eye(3)
        W[2, 2] = np.linalg.det(np.matmul(np.transpose(V), np.transpose(U)))

    # Caluclate the rotation matrix
    R = np.matmul(np.transpose(V), np.matmul(W, np.transpose(U)))

    # Calculate the translation from the rotated points
    rot_src_points = np.matmul(R, np.transpose(src_points))
    translation = tar_mean - np.mean(rot_src_points, 1)

    # Construct the affine matrix for use in PyCA
    if np.shape(src_points)[1] == 2:
        affine = np.zeros([3, 3])
        affine[0:2, 0:2] = R
        affine[0:2, 2] = translation
        affine[2, 2] = 1
    else:
        affine = np.zeros([4, 4])
        affine[0:3, 0:3] = R
        affine[0:3, 3] = translation
        affine[3, 3] = 1

    return affine