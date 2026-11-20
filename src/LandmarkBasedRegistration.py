"""
LandmarkBasedRegistration.py

LandMarker, 2023-2026
Michelle Kline, Department of Radiology and Imaging Sciences, University of Utah
Amanpreet Singh, Scientific Computing and Imaging Institute, University of Utah

Copyright (c) 2026 University of Utah
Licensed under the MIT License. See LICENSE file for details.

"""

from Common.try_aff import *
from LandmarkFunctions.SolveRigidMatrix import *
from LandmarkFunctions.applyMatrix2Grid import *
from LandmarkFunctions.TPS import *
from LandmarkFunctions.SplinesFlow import *
from Common.StructuredGridClass import *
from LandmarkFunctions.FFTSplines import *
from LandmarkFunctions.SolveAffineMatrix import *


def LandmarkBasedRegistration(fixedLandmarks : array.array, movingLandmarks : array.array,
                              movingVolume : array.array, method : tuple, device : str) -> tuple:
    """
    Perform landmark-based registration of a moving volume to a fixed coordinate system.
    Landmarks are assumed to be in voxel index coordinates of the corresponding image grids.

    This code implicitly assumes:
    fixedLandmarks and movingLandmarks are in the same physical units/scale (here: voxel indices).
    Distances between landmarks in both sets are comparable (i.e., voxel size is effectively the same in all directions
    and across images).
    The deformation field phi lives on the movingVolume’s grid and is defined in those same index units.
    Thus, any physical/world-space differences between moving and fixed images (spacing, orientation, origin) must be
    handled before calling this function.

    Args:
        fixedLandmarks: Target landmark coordinates (N x 3)
        movingLandmarks: Source landmark coordinates (N x 3)
        movingVolume: The moving image to register
        method: Tuple of (method_name, parameters_dict)
        device: PyTorch device ('cpu' or 'cuda')

    Returns:
        tuple: (phi, transformed_landmarks, rigid_matrix/affine) where phi is the deformation field,
               transformed_landmarks are the moving landmarks after transformation (N x 3), and
               rigid_matrix/affine is the transformation matrix (or None if not applicable).
               For FFTRigidThenSplinesFlow, returns affine instead of rigid_matrix.
    """

    movingVolume = torch.from_numpy(movingVolume.copy()).to(device)
    movingLandmarks = torch.from_numpy(movingLandmarks).to(device)
    fixedLandmarks = torch.from_numpy(fixedLandmarks).to(device)

    methodName, methodParameters = method

    # MMKline
    phi = None
    transformed_landmarks = None
    rigid_matrix = None

    if methodName not in ["Rigid", "Affine", "TPS", "SplinesFlow", "FFTSplinesFlow", "FFTRigidThenSplinesFlow"]:
        raise NotImplementedError

    else:
        # FIXME: why are we passing fixed, moving instead of moving, fixed? 12/2025

        if methodName == "Rigid":
            # Required Parameters : None
            rigid_matrix = torch.from_numpy(SolveRigidMatrix(fixedLandmarks, movingLandmarks)).to(device)
            phi = applyMatrix2Grid(movingVolume, rigid_matrix.float())
        elif methodName == "Affine":
            rigid_matrix = torch.from_numpy(SolveAffineMatrix(fixedLandmarks, movingLandmarks)).to(device)
            phi = applyMatrix2Grid(movingVolume, rigid_matrix.float())
        elif methodName == "TPS":
            # No rigid/affine matrix for TPS
            rigid_matrix = None
            # As of now Required Parameters : None
            phi = tps_3d(fixedLandmarks, movingLandmarks, movingVolume, device)
        elif methodName == "SplinesFlow":
            # FIXME: make parameters editable by user
            # # Required Parameters
            # flow_dict = ["kernel", "kernel_size", "nsteps", "df", "sigma"]
            #
            # # Check provided Parameters
            # for key in methodParameters.keys():
            #     if key not in flow_dict:
            #         print("Unexpected Parameters")
            #
            # # Set missing parameters if any
            # if len(methodParameters) == 5:
            #     print("Using Provided Parameters")
            # else:
            #     for key in methodParameters.keys():
            #         flow_dict.remove(key)
            #
            #     for key in flow_dict:
            #         if key == "kernel":
            #             methodParameters[key] = "Gaussian"
            #         elif key == "kernel_size":
            #             methodParameters[key] = 10
            #         elif key == "nsteps":
            #             methodParameters[key] = 10
            #         elif key == "df":
            #             methodParameters[key] = True
            #         elif key == "sigma":
            #             methodParameters[key] = 0
            #
            # # TODO Make kernel a functor
            #
            # kernel = methodParameters["kernel"]
            # kernel_size = methodParameters["kernel_size"]
            # nsteps = methodParameters["nsteps"]
            # df = methodParameters["df"]
            # sigma = methodParameters["sigma"]
            kernel = "Gaussian"
            kernel_size = 10
            nsteps = 10
            df = True
            sigma = 0

            phi = SplinesFlow(fixedLandmarks, movingLandmarks, movingVolume.shape, sigma,
                              kernel_size, df, nsteps, device)
            # No rigid/affine matrix for SplinesFlow
            rigid_matrix = None

        elif methodName == "FFTSplinesFlow":
            # FIXME: let user define parameters
            # # # Required Parameters
            # fft_dict = ["kernel", "kernel_size", "nsteps", "df"]
            # #
            # # # Check provided Parameters
            # for key in methodParameters.keys():
            #     if key not in fft_dict:
            #         print("Unexpected Parameters")
            # #
            # #         # Set missing parameters if any
            # if len(methodParameters) == 3:
            #     print("Using Provided Parameters")
            # else:
            #     for key in methodParameters.keys():
            #         fft_dict.remove(key)
            # #
            #     for key in fft_dict:
            #         if key == "kernel":
            #             methodParameters[key] = "MultiQuadratic"
            #         elif key == "kernel_size":
            #             methodParameters[key] = 1
            #         elif key == "nsteps":
            #             methodParameters[key] = 10
            # #
            # kernel = methodParameters["kernel"]
            # kernel_size = methodParameters["kernel_size"]
            # nsteps = methodParameters["nsteps"]
            kernel = "MultiQuadratic"
            kernel_size = 1
            nsteps = 10

            phi_init = eyeHField(movingVolume.shape, device)
            #
            source_vol = StructuredGrid(size=movingVolume.shape, tensor=movingVolume.unsqueeze(0))
            #
            phi = FFTSplines(movingLandmarks.t(), fixedLandmarks.t(), phi_init, source_vol, nsteps, kernel_size, device)
            # No rigid/affine matrix for FFTSplinesFlow
            rigid_matrix = None
        elif methodName == "FFTRigidThenSplinesFlow":
            # FIXME: let user define parameters
            # # # Required Parameters
            # fft_dict = ["kernel", "kernel_size", "nsteps", "df"]
            # #
            # # # Check provided Parameters
            # for key in methodParameters.keys():
            #     if key not in fft_dict:
            #         print("Unexpected Parameters")
            # #
            # #         # Set missing parameters if any
            # if len(methodParameters) == 3:
            #     print("Using Provided Parameters")
            # else:
            #     for key in methodParameters.keys():
            #         fft_dict.remove(key)
            #     #
            #     for key in fft_dict:
            #         if key == "kernel":
            #             methodParameters[key] = "MultiQuadratic"
            #         elif key == "kernel_size":
            #             methodParameters[key] = 1
            #         elif key == "nsteps":
            #             methodParameters[key] = 10
            # #
            # kernel = methodParameters["kernel"]
            # kernel_size = methodParameters["kernel_size"]
            # nsteps = methodParameters["nsteps"]
            kernel = "MultiQuadratic"
            kernel_size = 1
            nsteps = 10

            # start with Rigid Registration
            rigid_matrix = torch.from_numpy(SolveRigidMatrix(fixedLandmarks, movingLandmarks)).to(device)
            phi = applyMatrix2Grid(movingVolume, rigid_matrix.float())
            # now FFT Splines Flow
            R = rigid_matrix[:3, 0:3].to(device)
            t = rigid_matrix[:3, -1].to(device).unsqueeze(-1)

            tform_pts = torch.matmul(R.float(), movingLandmarks.t().float()) + t

            affine = torch.zeros(4, 4).to(device)
            affine[0:3, 0:3] = R
            affine[0:3, 3] = t.squeeze()
            affine[3, 3] = 1

            phi_affine = affine_transform(movingVolume.data.squeeze(), affine, movingVolume.shape)
            phi_affine[[0, 2], :, :, :] = phi_affine[[2, 0], :, :, :]

            phi_spline = eyeHField(movingVolume.shape, device)

            source_vol = StructuredGrid(size=movingVolume.shape, tensor=movingVolume.unsqueeze(0))

            phi_spline = FFTSplines(tform_pts, fixedLandmarks.t(), phi_spline, source_vol, nsteps, kernel_size, device)

            phi = composeHfields(phi_spline, phi_affine, movingVolume.shape, 3)
            # For FFTRigidThenSplinesFlow, return affine instead of rigid_matrix
            rigid_matrix = affine

        # MMKline - moved this interp() call to outside of this method. So this method now only returns the dfield (and
        # registered lms?).
        # reg_vol = interp(movingVolume, phi)
        # reg_vol = reg_vol.cpu().detach().numpy()

    # # Compute transformed (registered) moving landmarks using the deformation field
    # def tformPts(_pts, _phi):
    #     new_pts = torch.zeros_like(_pts)
    #     for x in range(_pts.shape[0]):
    #         new_pts[x] = interp(_phi, _pts[x].flip(-1).unsqueeze(1).float(), sample_pts=True).squeeze().flip(-1)
    #
    #     return new_pts

    # if phi is not None:
        # TODO: Compute phi_inv (inverse of phi)
        # For now, using phi - need to implement inverse computation
        # phi_inv = phi  # Placeholder - replace with actual inverse computation
        # transformed_landmarks = transformPoints(movingLandmarks.float(), phi_inv)
        # tfpts = transformPoints(movingLandmarks.float(), phi)

    # MMKline
    del methodParameters
    # Convert rigid_matrix/affine to numpy if it exists, otherwise return None
    matrix_result = None
    if rigid_matrix is not None:
        matrix_result = rigid_matrix.cpu().detach().numpy()
    return phi, matrix_result
