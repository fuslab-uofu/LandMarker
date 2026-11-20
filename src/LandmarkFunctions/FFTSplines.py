from Common.transformPoints import *
from Common.Kernels import *
from Common.composeHfields import *
import math


def FFTSplines(src_pts, tar_pts, phi, src_im, niters, kernel_SIGMA, device):

    energy = []

    num_landmarks = src_pts.shape[1]

    src_pts_t = src_pts
    tar_pts_t = tar_pts

    energy.append(torch.sqrt(((tar_pts - src_pts_t) ** 2).sum(0)).mean().item())


    print(energy[-1])
    for iters in range(1, niters+1):
        K = torch.zeros(num_landmarks, num_landmarks).to(device)

        for r in range(num_landmarks):
            for c in range(num_landmarks):
                # K[r, c] = torch.exp(- ((tar_pts[0, r] - tar_pts[0, c])**2 +
                #                        (tar_pts[1, r] - tar_pts[1, c])**2 +
                #                        (tar_pts[2, r] - tar_pts[2, c])**2)
                #                         /(2 * SIGMA**2))
                K[r, c] = MultiQuadRBF_Kernel(tar_pts[:, r], tar_pts[:, c], kernel_SIGMA)

        d = (tar_pts - src_pts_t)/(niters + 1 - iters)
        b = torch.matmul(torch.inverse(K), d.t().float()).t()

        v = torch.zeros_like(phi)

        dx = phi[2].squeeze()
        dy = phi[1].squeeze()
        dz = phi[0].squeeze()

        for r in range(num_landmarks):
            # temp = torch.exp(- ((dx - tar_pts[0, r])**2 +
            #                     (dy - tar_pts[1, r])**2 +
            #                     (dz - tar_pts[2, r])**2)
            #                     /(2 * SIGMA**2))
            temp = MultiQuadRBF_Kernel(phi, tar_pts[:, r], kernel_SIGMA, deformation=True)
            tempx = b[0, r] * temp
            tempy = b[1, r] * temp
            tempz = b[2, r] * temp
            v[0] = v[0] + tempx
            v[1] = v[1] + tempy
            v[2] = v[2] + tempz

        v[[0, 2], :, :, :] = v[[2, 0], :, :, :]
        v = div_free_projection(v, src_im, device)
        v[[0, 2], :, :, :] = v[[2, 0], :, :, :]

        phi[2] = dx + v[0]
        phi[1] = dy + v[1]
        phi[0] = dz + v[2]

        # hx = phi[1]
        # hy = phi[2]

        # plt.figure()
        # plt.plot(hy.cpu().numpy()[0::2, 0::2, 64], hx.cpu().numpy()[0::2, 0::2, 64], 'k')
        # plt.plot(np.transpose(hy.cpu().numpy()[0::2, 0::2, 64]), np.transpose(hx.cpu().numpy()[0::2, 0::2, 64]), 'k')
        # plt.plot(src_pts[1, :].cpu().numpy(), src_pts[0, :].cpu().numpy(), 'r*')
        # plt.plot(tar_pts[1, :].cpu().numpy(), tar_pts[0, :].cpu().numpy(), 'bo')
        # plt.plot(src_pts_t[1, :].cpu().numpy(), src_pts_t[0, :].cpu().numpy(), 'gs')
        # plt.show()

        # MMKline - bugfix. this method calls interp(), which requires device, so added device to arguments
        src_pts_t = transformPoints(torch.tensor(src_pts.t()), phi).t()

        energy.append(torch.sqrt(((tar_pts - src_pts_t) ** 2).sum(0)).mean().item())

        if energy[-1] < 2:
            break

        print(energy[-1])

    return phi


def div_free_projection(x, grid, device):
    # size = grid.size.view(*grid.size.shape, *([1] * len(grid.size)))  # MMKline - added device as parameter
    size = grid.size.view(*grid.size.shape, *([1] * len(grid.size))).to(device)
    vecs = [torch.arange(0, x, device=device).float() for x in grid.size]

    grids = torch.meshgrid(vecs, indexing='ij')
    grids = torch.stack(grids, 0)

    # Create sin grids
    # Compute the cosine grids
    cos_grids = (2 * torch.cos((2.0 * math.pi * grids) / size) - 2)
    sin_grids = torch.sin((2 * math.pi * grids) / size)

    # Make the spatial dimensions linear
    cos_grids = cos_grids.view(len(grid.size), -1).permute(1, 0)
    sin_grids = sin_grids.view(len(grid.size), -1).permute(1, 0)

    nsq = (sin_grids ** 2).sum(0)

    fft = torch.fft.fftn(x.data, norm="backward")
    fft = torch.view_as_real(fft)
    real, imag = fft.split(1, -1)

    # We need to linearize the matrices so we can easily do the multiplication
    real = real.squeeze().reshape(len(x.size()) - 1, -1).permute(1, 0)
    imag = imag.squeeze().reshape(len(x.size()) - 1, -1).permute(1, 0)

    real_dot = (real * sin_grids).sum(-1, keepdim=True)
    imag_dot = (imag * sin_grids).sum(-1, keepdim=True)

    real_app = real - (real_dot * sin_grids / nsq)
    imag_app = imag - (imag_dot * sin_grids / nsq)

    # Deal with the nan
    real_app[real_app != real_app] = real.squeeze()[real_app != real_app]
    imag_app[imag_app != imag_app] = imag.squeeze()[imag_app != imag_app]

    # Return tensors to their actual shapes
    real_app = real_app.permute(1, 0).view(len(x.size()) - 1, *x.size()[1:])
    imag_app = imag_app.permute(1, 0).view(len(x.size()) - 1, *x.size()[1:])
    smooth_fft = torch.stack((real_app, imag_app), -1)

    # Inverse fourier transform
    # out_tensor = torch.irfft(smooth_fft, signal_ndim=self.dim, normalized=False, onesided=False)
    out_tensor = torch.fft.ifftn(torch.view_as_complex(smooth_fft), norm="backward").real

    return out_tensor
