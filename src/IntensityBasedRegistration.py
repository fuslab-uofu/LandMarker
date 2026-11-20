"""
IntensityBasedRegistration.py

LandMarker, 2023-2026
Michelle Kline, Department of Radiology and Imaging Sciences, University of Utah
Amanpreet Singh, Scientific Computing and Imaging Institute, University of Utah

Copyright (c) 2026 University of Utah
Licensed under the MIT License. See LICENSE file for details.

"""
import array
from IntensityFunctions.sdiff import sdiff


class RegistrationCancelled(Exception):
    """Raised when the user cancels a long-running registration."""
    pass
from Common.varianceEqualisation import *
from copy import deepcopy
from Common.UpsaclePhi import *
from Common.composeHfields import *
from IntensityFunctions.elastic import elastic
from IntensityFunctions.Diffeo import diffeomorphic


def IntensityBasedRegistration(fixedVolume : array.array, movingVolume : array.array,
                               method : tuple, device : str,
                               cancel_check=None, use_mask=True) -> tuple:  # J.Bonaventura
    """
    # TODO Add comments
    :param cancel_check: Optional callable with no args. If provided and returns True, registration is cancelled.
	# J. Bonaventura
    :param use_mask: If True (default), sdiff masks the difference image to non-zero fixed voxels. Set False to
                     compute the gradient over the full volume — useful when variance equalisation produces
                     negative values that the mask would incorrectly exclude.
    """
    def check_cancel():
        if cancel_check and cancel_check():
            raise RegistrationCancelled()

    fixedVolume = torch.from_numpy(fixedVolume.copy()).to(device)
    movingVolume = torch.from_numpy(movingVolume.copy()).to(device)

    # Do Variance Equalisation
    fixedVolume_vareq = varEq(fixedVolume.unsqueeze(-1), sigma=1, device=device).squeeze()
    movingVolume_vareq = varEq(movingVolume.unsqueeze(-1), sigma=1, device=device).squeeze()

    fixedVolume_vareq = fixedVolume_vareq/fixedVolume_vareq.max()
    movingVolume_vareq = movingVolume_vareq/movingVolume_vareq.max()

    methodName, methodParameters = method

    # MMKline
    phi = None
    energies = None

    if methodName not in ["Rigid", "Affine", "Elastic", "Diffeomorphic", "LDMM", "sdiff"]:
        raise NotImplementedError

    else:
        if methodName == "Rigid":
            raise NotImplementedError
            # print("TO DO")
        elif methodName == "Affine":
            raise NotImplementedError
            # print("TO DO")
        elif methodName == "Elastic":
            # Required Parameters
            elastic_dict = ["sigma", "lambda", "eps", "niters"]

            for key in methodParameters.keys():
                if key not in elastic_dict:
                    print("Unexpected Parameters")

            if len(methodParameters) == 4:
                print("Using Provided Parameters")
            else:
                for key in methodParameters.keys():
                    elastic_dict.remove(key)

                for key in elastic_dict:
                    if key == "sigma":
                        methodParameters[key] = [500]*3
                    elif key == "lambda":
                        methodParameters[key] = [1] * 3
                    elif key == "eps":
                        methodParameters[key] = [100]*3  # [10000, 10000, 10000]
                    elif key == "niters":
                        methodParameters[key] = [10]*3  # [10000, 5000, 1000]

            sigma = methodParameters["sigma"]
            l = methodParameters["lambda"]
            eps = methodParameters["eps"]
            niters = methodParameters["niters"]

            factor = 2 ** (len(niters) - 1)

            moving = torch.from_numpy(
                rescale(movingVolume_vareq.cpu().numpy().squeeze(), 1 / factor, anti_aliasing=True)).to(
                device)

            phi_hist = []

            for x in range(len(niters)):
                check_cancel()
                fixed = torch.from_numpy(
                    rescale(fixedVolume_vareq.cpu().numpy().squeeze(), 1 / factor, anti_aliasing=True)).to(device)

                phi, energies = elastic(moving, fixed, sigma[x], l[x], eps[x], niters[x],
                                       cancel_check=cancel_check, cancel_exception=RegistrationCancelled)
                phi_hist.append(phi)

                if x != (len(niters) - 1):
                    factor /= 2
                    im_next = torch.from_numpy(
                        rescale(movingVolume_vareq.cpu().numpy().squeeze(), 1 / factor, anti_aliasing=True)).to(device)

                    phi_up = ScaleList(deepcopy(phi_hist), device)
                    moving = interp(deepcopy(im_next).to(device), phi_up)

            final_deformation = composeHfields(phi_hist[-1], ScaleList(deepcopy(phi_hist[0:2]), device), None, None)

            phi = final_deformation

        elif methodName == "Diffeomorphic":
            # Required Parameters
            diffeo_dict = ["sigma", "lambda", "eps", "niters"]

            for key in methodParameters.keys():
                if key not in diffeo_dict:
                    print("Unexpected Parameters")

            if len(methodParameters) == 4:
                print("Using Provided Parameters")
            else:
                for key in methodParameters.keys():
                    diffeo_dict.remove(key)

                for key in diffeo_dict:
                    if key == "sigma":
                        methodParameters[key] = [500] * 3
                    elif key == "lambda":
                        methodParameters[key] = [1] * 3
                    elif key == "eps":
                        methodParameters[key] = [100]*3  # [10000, 10000, 10000]
                    elif key == "niters":
                        methodParameters[key] = [10]*3  # [10000, 5000, 1000]

            sigma = methodParameters["sigma"]
            l = methodParameters["lambda"]
            eps = methodParameters["eps"]
            niters = methodParameters["niters"]

            factor = 2 ** (len(niters) - 1)

            moving = torch.from_numpy(
                rescale(movingVolume_vareq.cpu().numpy().squeeze(), 1 / factor, anti_aliasing=True)).to(
                device)

            phi_hist = []

            for x in range(len(niters)):
                check_cancel()
                fixed = torch.from_numpy(
                    rescale(fixedVolume_vareq.cpu().numpy().squeeze(), 1 / factor, anti_aliasing=True)).to(device)

                phi, energies = diffeomorphic(moving, fixed, sigma[x], l[x], eps[x], niters[x],
                                              cancel_check=cancel_check, cancel_exception=RegistrationCancelled)
                phi_hist.append(phi)

                if x != (len(niters) - 1):
                    factor /= 2
                    im_next = torch.from_numpy(
                        rescale(movingVolume_vareq.cpu().numpy().squeeze(), 1 / factor, anti_aliasing=True)).to(device)

                    phi_up = ScaleList(deepcopy(phi_hist), device)
                    moving = interp(deepcopy(im_next).to(device), phi_up)

            final_deformation = composeHfields(phi_hist[-1], ScaleList(deepcopy(phi_hist[0:2]), device), None, None)

            phi = final_deformation
        elif methodName == "FFT":
            raise NotImplementedError
            # print("TO DO")
        elif methodName == "LDMM":
            raise NotImplementedError
            # print("TO DO")
        elif methodName == "sdiff":
            # Required Parameters
            sdiff_dict = ["sigma", "eps", "niters"]

            # Check provided Parameters
            for key in methodParameters.keys():
                if key not in sdiff_dict:
                    print("Unexpected Parameters")

            # Set missing parameters if any
            if len(methodParameters) == 3:
                print("Using Provided Parameters")
            else:
                for key in methodParameters.keys():
                    sdiff_dict.remove(key)

                for key in sdiff_dict:
                    if key == "sigma":
                        methodParameters[key] = [1]*3
                    elif key == "eps":
                        methodParameters[key] = [1]*3
                    elif key == "niters":
                        methodParameters[key] = [10]*3

            # Now Call the registration function
            sigma = methodParameters["sigma"]
            eps = methodParameters["eps"]
            niters = methodParameters["niters"]

            factor = 2 ** (len(niters) - 1)

            moving = torch.from_numpy(
                rescale(movingVolume_vareq.cpu().numpy().squeeze(), 1 / factor, anti_aliasing=True)).to(
                device)
			# J. Bonaventura
            # Use actual rescaled shape for det_im — integer division can differ from
            # skimage's rounding when dimensions are not divisible by factor
            det_im = torch.ones_like(moving)

            phi_hist = []

            for x in range(len(niters)):
                check_cancel()
                fixed = torch.from_numpy(
                    rescale(fixedVolume_vareq.cpu().numpy().squeeze(), 1 / factor, anti_aliasing=True)).to(device)
				# J. Bonaventura
                # Crop to minimum shape per dimension — ScaleList upsample can produce
                # a shape that's 1 voxel smaller than a fresh rescale of the same volume,
                # causing a mask shape mismatch in sdiff. Cropping from the end is safe
                # (no pixel shift).
                crop = tuple(slice(0, min(m, f)) for m, f in zip(moving.shape, fixed.shape))
                moving = moving[crop]
                fixed  = fixed[crop]
                det_im = det_im[crop]

                phi, phi_inv, det_im, energies = sdiff(moving, fixed, sigma[x], eps[x], niters[x], det_im, device,
                                                       cancel_check=cancel_check, cancel_exception=RegistrationCancelled,
                                                       use_mask=use_mask)  # J. Bonaventura
                phi_hist.append(phi)

                if x != (len(niters) - 1):
                    factor /= 2
                    im_next = torch.from_numpy(
                        rescale(movingVolume_vareq.cpu().numpy().squeeze(), 1 / factor, anti_aliasing=True)).to(device)

                    phi_up = ScaleList(deepcopy(phi_hist), device)
                    moving = interp(deepcopy(im_next).to(device), phi_up)
                    det_im = interp(deepcopy(det_im).to(device), phi_up)

                    det_im[det_im == 0] = 1

            final_deformation = composeHfields(phi_hist[-1], ScaleList(deepcopy(phi_hist[0:2]), device), None, None)

            phi = final_deformation

        # MMKline pulled this call to interp out, so this function does one thing: calculates phi
        # reg_vol = interp(movingVolume, phi)
        # reg_vol = reg_vol.cpu().detach().numpy()

    # MMKline
    # del methodParameters
    return phi, energies


