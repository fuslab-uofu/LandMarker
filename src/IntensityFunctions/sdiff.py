import torch
import numpy as np
import time
from Common.imageGradient import *
from Common.deformImage import *
from Common.jacobianHField import *
from Common.composeHfields import *
from Common.eyeHField import *
import math
import copy
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
plt.ion()
from PIL import Image


# J. Bonaventura
def sdiff(I0, I1, sigma, eps, niters, det_phi, device='cpu', cancel_check=None, cancel_exception=None, use_mask=True,
          trace_voxel=None, trace_log=None):
    iota = torch.view_as_complex(torch.stack((torch.tensor(0).float(), torch.tensor(1).float()), -1)).to(device)
    s = list(I0.shape)
    dims = len(s)

    # Create Identity Fields
    I = eyeHField(s, device)

    phi = copy.deepcopy(I)
    pPhi = copy.deepcopy(I)
    sPhi = copy.deepcopy(I)
    phi_inv = copy.deepcopy(I)

    # Flag for divergence free
    sDiff = 1

    # Set up the filter for Inverse Laplacian
    L = np.zeros(s)

    if dims == 2:
        for u in range(s[0]):
            for v in range(s[1]):
                L[u, v] = -2.0 * (math.cos(2 * math.pi * u / s[0]) + math.cos(2 * math.pi * v / s[1])) + 4

        L[0, 0] = 1
        Linv = 1/L
        L[0, 0] = 0

    elif dims == 3:
        for u in range(s[0]):
            for v in range(s[1]):
                for w in range(s[2]):
                    L[u, v, w] = -2.0 * (math.cos(2 * math.pi * u / s[0]) + math.cos(2 * math.pi * v / s[1]) +
                                         math.cos(2 * math.pi * w / s[2])) + 6.0

        L[0, 0, 0] = 1
        Linv = 1 / L
        L[0, 0, 0] = 0

    else:
        raise NotImplementedError

    L = torch.from_numpy(L).to(device)
    Linv = torch.from_numpy(Linv).to(device)

    # # Start with detDphi as ones and create an energy list

    detDphi = det_phi.to(device)
    l2_energy_list = []
    regularisation_energy_list = []
    total_energy_list = []

    # Create sinWX and sinWY needed later in computation
    sinWX = np.sin(2 * math.pi * np.arange(0, s[0]) / s[0])
    sinWY = np.sin(2 * math.pi * np.arange(0, s[1]).reshape(-1, 1) / s[1])

    sinWX = torch.from_numpy(sinWX).to(device)
    sinWY = torch.from_numpy(sinWY).to(device)
    sinWX = sinWX.repeat(s[1], 1).t()
    sinWY = sinWY.repeat(1, s[0]).t()

    if dims==3:
        sinWX = sinWX.unsqueeze(-1).repeat(1, 1, s[2])
        sinWY = sinWY.unsqueeze(-1).repeat(1, 1, s[2])

        # Create sinWZ
        sinWZ = np.sin(2 * math.pi * np.arange(0, s[2]) / s[2])
        sinWZ = torch.from_numpy(sinWZ).to(device)
        sinWZ = sinWZ.repeat(s[1], 1)
        sinWZ = sinWZ.unsqueeze(0).repeat(s[0], 1, 1)

        sinWX , sinWZ = sinWZ, sinWX

    nsq = sinWX**2 + sinWY**2

    if dims == 3:
        nsq += sinWZ**2

# J. Bonaventura begin ---
    if use_mask:
        mask = torch.zeros_like(I1)
        mask[I1 > 0] = 1
        mask[I0 == 0] = 0
    else:
        mask = torch.ones_like(I1)

    # TEMPORARY diagnostic instrumentation for the boundary-collapse investigation --
    # remove once that's resolved. Records the given voxel's coordinate at each of
    # the compose steps that feed into phi, so we can see exactly where/when it
    # drifts out of [1, s] and gets hard-zeroed by interp()'s checkBounds.
    def _trace(stage):
        if trace_voxel is not None and trace_log is not None:
            ti, tj, tk = trace_voxel
            trace_log.append({'iter': iter, 'stage': stage, 'Vp': Vp[:, ti, tj, tk].tolist(),
                               'pPhi': pPhi[:, ti, tj, tk].tolist(), 'sPhi': sPhi[:, ti, tj, tk].tolist(),
                               'phi': phi[:, ti, tj, tk].tolist()})
# J. Bonaventura end ---

    # # Start Iterating
    for iter in range(1, niters+1):
        if cancel_check and cancel_exception and cancel_check():
            raise cancel_exception()

        iterStartTime = time.time()
        print("Iteration : ", iter)

        # Compute the Jacobian of phi:
        Dphi = jacobianHField(phi)

        # Compute the square root
        sdetDphi = torch.sqrt(detDphi)

        # Compute the Laplacian
        df = torch.fft.fftn(sdetDphi)
        if dims == 2:
            real_idf = torch.view_as_real(torch.fft.ifftn(df * L))[:, :, 0]
        elif dims == 3:
            real_idf = torch.view_as_real(torch.fft.ifftn(df * L))[:, :, :, 0]

        LsdetDphi = 4 * real_idf / np.sqrt(s[0])

        # Deform I0 by phi

        J = deformImage(I0, phi, s, dims)

        Diff = (J - I1) * mask

        l2_energy = (Diff**2).data.mean()
        reg_energy = sigma * ((sdetDphi - 1)**2).mean()

        l2_energy_list.append(l2_energy.item())
        regularisation_energy_list.append(reg_energy.item())
        total_energy_list.append((l2_energy + reg_energy).item())

        # Compute the gradient
        G = imageGradient(J.data.squeeze())

        Gx = Diff.data.squeeze() * G[0]
        Gy = Diff.data.squeeze() * G[1]

        if dims == 3:
            Gz = Diff.data.squeeze() * G[2]

        # Now take the divergence free part
        fftx = torch.fft.fftn(Gx)
        ffty = torch.fft.fftn(Gy)

        if dims == 3:
            fftz = torch.fft.fftn(Gz)

        # Smooth them by Linv
        fftxL = fftx * Linv
        fftyL = ffty * Linv

        if dims == 3:
            fftzL = fftz * Linv


        divGF = torch.zeros(s, dtype=torch.complex64).to(device)

        loopStartTime = time.time()

        bRealX = torch.view_as_real(fftxL).split(1, -1)[0].squeeze()
        bRealY = torch.view_as_real(fftyL).split(1, -1)[0].squeeze()
        bImagX = torch.view_as_real(fftxL).split(1, -1)[1].squeeze()
        bImagY = torch.view_as_real(fftyL).split(1, -1)[1].squeeze()

        if dims == 3:
            bRealZ = torch.view_as_real(fftzL).split(1, -1)[0].squeeze()
            bImagZ = torch.view_as_real(fftzL).split(1, -1)[1].squeeze()

        # Real part of S dot V in Fourier
        reSdotV = bRealX * sinWX + bRealY * sinWY

        # Imag part of S dot V in Fourier
        imSdotV = bImagX * sinWX + bImagY * sinWY

        if dims == 3:
            reSdotV += bRealZ * sinWZ
            imSdotV += bImagZ * sinWZ

        # subtract S dot V
        vRealX = bRealX - reSdotV*sinWX/nsq
        vRealY = bRealY - reSdotV*sinWY/nsq

        vImagX = bImagX - imSdotV*sinWX/nsq
        vImagY = bImagY - imSdotV*sinWY/nsq

        if dims == 3:
            vRealZ = bRealZ - reSdotV*sinWZ/nsq
            vImagZ = bImagZ - imSdotV*sinWZ/nsq  # J. Bonaventura

        fftxL = torch.view_as_complex(torch.stack((vRealX, vImagX), -1))
        fftyL = torch.view_as_complex(torch.stack((vRealY, vImagY), -1))

        if dims == 3:
            fftzL = torch.view_as_complex(torch.stack((vRealZ, vImagZ), -1))

        if dims == 2:
            divGF = iota*(fftx * sinWX + ffty * sinWY)
            divGF[0, 0] = 0
            fftxL[0, 0] = 0
            fftyL[0, 0] = 0

        elif dims == 3:
            divGF = iota * (fftx * sinWX + ffty * sinWY + fftz * sinWZ)
            divGF[0, 0, 0] = 0
            fftxL[0, 0, 0] = 0
            fftyL[0, 0, 0] = 0
            fftzL[0, 0, 0] = 0

        loopEndTime = time.time()
        # print("Time taken for loop : ", loopEndTime - loopStartTime, "s")

        divfx = torch.view_as_real(torch.fft.ifftn(fftxL)).split(1, -1)[0].squeeze()
        divfy = torch.view_as_real(torch.fft.ifftn(fftyL)).split(1, -1)[0].squeeze()
        divG = torch.view_as_real(torch.fft.ifftn(divGF)).split(1, -1)[0].squeeze()

        if dims == 3:
            divfz = torch.view_as_real(torch.fft.ifftn(fftzL)).split(1, -1)[0].squeeze()
        # Compute G

        g = divG + 4 * sigma * LsdetDphi

        # Now for the sobolev gradient filter g

        gf = torch.fft.fftn(g)
        g = torch.view_as_real(torch.fft.ifftn(gf*Linv)).split(1, -1)[0].squeeze()

        gf = gf * Linv * Linv

        f = torch.view_as_real(torch.fft.ifftn(gf)).split(1, -1)[0].squeeze()

        # v is simply grad f

        v = imageGradient(f)

        Vp = I + eps*v
        if dims == 2:
            Vp[0, :, :][Vp[0, :, :] > s[0]] = s[0]
            Vp[1, :, :][Vp[1, :, :] > s[1]] = s[1]
            Vp[Vp < 1] = 1

        elif dims == 3:
            Vp[0, :, :, :][Vp[0, :, :, :] > s[2]] = s[2]
            Vp[1, :, :, :][Vp[1, :, :, :] > s[1]] = s[1]
            Vp[2, :, :, :][Vp[2, :, :, :] > s[0]] = s[0]
            Vp[Vp < 1] = 1

        # Compose fields
        _trace('potential_Vp_clamped')

        pPhi = composeHfields(pPhi, Vp, s, dims)
        _trace('pPhi_after_compose')

        Vp[0] = -divfx
        Vp[1] = -divfy

        if dims == 3:
            Vp[2] = -divfz

        Vp = I + eps*Vp
        if dims == 2:
            Vp[0, :, :][Vp[0, :, :] > s[0]] = s[0]
            Vp[1, :, :][Vp[1, :, :] > s[1]] = s[1]
            Vp[Vp < 1] = 1

        elif dims == 3:
            Vp[0, :, :, :][Vp[0, :, :, :] > s[2]] = s[2]
            Vp[1, :, :, :][Vp[1, :, :, :] > s[1]] = s[1]
            Vp[2, :, :, :][Vp[2, :, :, :] > s[0]] = s[0]
            Vp[Vp < 1] = 1

        _trace('solenoidal_Vp_clamped')  # J. Bonaventura
        sPhi = composeHfields(sPhi, Vp, s, dims)
        _trace('sPhi_after_compose')  # J. Bonaventura

        # Now add div free part
        if sDiff:
            v[0] = v[0] - divfx
            v[1] = v[1] - divfy

            if dims == 3:
                v[2] = v[2] - divfz

        Vp = I + eps * v
        if dims == 2:
            Vp[0, :, :][Vp[0, :, :] > s[0]] = s[0]
            Vp[1, :, :][Vp[1, :, :] > s[1]] = s[1]
            Vp[Vp < 1] = 1

        elif dims == 3:
            Vp[0, :, :, :][Vp[0, :, :, :] > s[2]] = s[2]
            Vp[1, :, :, :][Vp[1, :, :, :] > s[1]] = s[1]
            Vp[2, :, :, :][Vp[2, :, :, :] > s[0]] = s[0]
            Vp[Vp < 1] = 1

        phi = composeHfields(pPhi, sPhi, s, dims)
        _trace('phi_after_final_compose')  # J. Bonaventura

        v_phi_inv = composeHfields(v, phi_inv, s, dims)
        phi_inv = phi_inv - eps * v_phi_inv

        # Deform detDphi by v
        detDphi = deformImage(detDphi.float(), Vp, s, dims) * torch.exp(-eps*g)


        iterEndTime = time.time()
        # print("Time taken for iteration : ", iterEndTime - iterStartTime, "s")
        print(f"Matching Energy at iteration {iter} is {l2_energy_list[iter-1]}")
        print(f"Reg Energy at iteration {iter} is {regularisation_energy_list[iter-1]}")
        print(f"Total Energy at iteration {iter} is {total_energy_list[iter-1]}")


    energies = [l2_energy_list, regularisation_energy_list, total_energy_list]
    # print("Registration Complete")

    return phi, phi_inv, detDphi, energies
