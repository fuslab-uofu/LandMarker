import torch
import numpy as np


def SolveAffineMatrix(source_landmarks, target_landmarks):

    source_landmarks = source_landmarks.float()
    target_landmarks = target_landmarks.float()

    source_landmarks_centered = source_landmarks - source_landmarks.mean(0)
    target_landmarks_centered = target_landmarks - target_landmarks.mean(0)

    # Solve for the transform between the points
    R = torch.matmul(
        torch.matmul(
            target_landmarks_centered.t(), source_landmarks_centered
        ),
        torch.matmul(
            source_landmarks_centered.t(), source_landmarks_centered
        ).inverse()
    )

    # Solve for the translation
    translation = target_landmarks.mean(0) - torch.matmul(R, source_landmarks.mean(0).t()).t()

    affine = torch.zeros([4, 4])
    affine[0:3, 0:3] = R
    affine[0:3, 3] = translation
    affine[3, 3] = 1

    return affine.cpu().numpy()