import torch

from Common.eyeHField import *
from LandmarkFunctions.GaussianKernel import *
from LandmarkFunctions.Update import *
from tqdm import tqdm


def SplinesFlow(src_pts, tar_pts, size, sigma, k_sigma, df, NumT, device='cpu'):

    num_landmarks = src_pts.shape[0]
    dim = src_pts.shape[1]
    id = eyeHField(size)

    result_size = src_pts.shape[0] * src_pts.shape[1]
    X = np.empty(result_size)
    Y = np.empty(result_size)

    count = 0

    for j in range(num_landmarks):
        for i in range(dim):
            X[count] = src_pts[j, i]
            Y[count] = tar_pts[j, i]
            count += 1

    D = X - Y

    # Now generate the flow

    disp = torch.from_numpy(D / NumT).float().to(device)

    disp_reshaped = torch.zeros_like(src_pts)

    count = 0
    for j in range(num_landmarks):
        for i in range(dim):
            disp_reshaped[j, i] = disp[count]
            count += 1

    K = torch.zeros(num_landmarks * dim, num_landmarks * dim)
    tar_pts_t = tar_pts
    deformation = id.to(device)
    deformation[[0, 2], :, :, :] = deformation[[2, 0], :, :, :]

    for k in tqdm(range(NumT)):

        for r in range(num_landmarks):
            for c in range(num_landmarks):
                K[r * dim:((r + 1) * dim), c * dim:((c + 1) * dim)] = GaussianKernel(tar_pts_t[r, :], tar_pts_t[c, :],
                                                                                     k_sigma, df, device=device)

        K = K.to(device)

        B = torch.matmul(torch.linalg.inv(K + (sigma ** 2 * torch.eye(num_landmarks * dim).to(device))), disp.unsqueeze(-1))

        temp_def = torch.zeros_like(deformation).float().to(device)

        for n in range(num_landmarks):
            temp_def += update_def(B[n * dim:(n + 1) * dim].t(), tar_pts_t[n, :],
                                   deformation, k_sigma, df, device=device)

        deformation = deformation + temp_def

    deformation[[0, 2], :, :, :] = deformation[[2, 0], :, :, :]

    # print(temp_def.max())
    return deformation.to(device)

