from Common.Interp import *


def normalize(phi, m):
    phi = phi - 1
    phi = phi - (m-1)/2
    phi = phi/((m-1)/2)
    return phi


def deformImage(f, phi, s, dim):
    new_im = interp(f.squeeze(), phi)
    return new_im

