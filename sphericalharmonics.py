import torch
import math
import numpy as np

'''
code taken and adapted from pyredner
'''

# Code adapted from "Spherical Harmonic Lighting: The Gritty Details", Robin Green
# http://silviojemma.com/public/papers/lighting/spherical-harmonic-lighting.pdf
class SphericalHarmonics:
    def __init__(self, envMapResolution, device):
        self.device = device
        self.setEnvironmentMapResolution(envMapResolution)

    def setEnvironmentMapResolution(self, res):
        res = (res, res)
        self.resolution = res
        uv = np.mgrid[0:res[1], 0:res[0]].astype(np.float32)
        self.theta = torch.from_numpy((math.pi / res[1]) * (uv[1, :, :] + 0.5)).to(self.device)
        self.phi = torch.from_numpy((2 * math.pi / res[0]) * (uv[0, :, :] + 0.5)).to(self.device)

    def smoothSH(self, coeffs, window=6):
        ''' multiply (convolve in sptial domain) the coefficients with a low pass filter.
        Following the recommendation in https://www.ppsloan.org/publications/shdering.pdf
        '''
        smoothed_coeffs = torch.zeros_like(coeffs)
        smoothed_coeffs[:, 0] += coeffs[:, 0]
        smoothed_coeffs[:, 1:1 + 3] += \
            coeffs[:, 1:1 + 3] * math.pow(math.sin(math.pi * 1.0 / window) / (math.pi * 1.0 / window), 4.0)
        smoothed_coeffs[:, 4:4 + 5] += \
            coeffs[:, 4:4 + 5] * math.pow(math.sin(math.pi * 2.0 / window) / (math.pi * 2.0 / window), 4.0)
        smoothed_coeffs[:, 9:9 + 7] += \
            coeffs[:, 9:9 + 7] * math.pow(math.sin(math.pi * 3.0 / window) / (math.pi * 3.0 / window), 4.0)
        return smoothed_coeffs


    def associatedLegendrePolynomial(self, l, m, x):
        pmm = torch.ones_like(x)
        if m > 0:
            somx2 = torch.sqrt((1 - x) * (1 + x))
            fact = 1.0
            for i in range(1, m + 1):
                pmm = pmm * (-fact) * somx2
                fact += 2.0
        if l == m:
            return pmm
        pmmp1 = x * (2.0 * m + 1.0) * pmm
        if l == m + 1:
            return pmmp1
        pll = torch.zeros_like(x)
        for ll in range(m + 2, l + 1):
            pll = ((2.0 * ll - 1.0) * x * pmmp1 - (ll + m - 1.0) * pmm) / (ll - m)
            pmm = pmmp1
            pmmp1 = pll
        return pll


    def normlizeSH(self, l, m):
        return math.sqrt((2.0 * l + 1.0) * math.factorial(l - m) / \
                         (4 * math.pi * math.factorial(l + m)))

    def SH(self, l, m, theta, phi):
        if m == 0:
            return self.normlizeSH(l, m) * self.associatedLegendrePolynomial(l, m, torch.cos(theta))
        elif m > 0:
            return math.sqrt(2.0) * self.normlizeSH(l, m) * \
                   torch.cos(m * phi) * self.associatedLegendrePolynomial(l, m, torch.cos(theta))
        else:
            return math.sqrt(2.0) * self.normlizeSH(l, -m) * \
                   torch.sin(-m * phi) * self.associatedLegendrePolynomial(l, -m, torch.cos(theta))

    def toEnvMap(self, shCoeffs, smooth = False):
        '''
        create an environment map from given sh coeffs
        :param shCoeffs: float tensor [n, bands * bands, 3]
        :param smooth: if True, the first 3 bands are smoothed
        :return: environment map tensor [n, resX, resY, 3]
        '''
        assert(shCoeffs.dim() == 3 and shCoeffs.shape[-1] == 3)
        envMaps = torch.zeros( [shCoeffs.shape[0], self.resolution[0], self.resolution[1], 3]).to(shCoeffs.device)
        for i in range(shCoeffs.shape[0]):
            envMap =self.constructEnvMapFromSHCoeffs(shCoeffs[i], smooth)
            envMaps[i] = envMap
        return envMaps
    def constructEnvMapFromSHCoeffs(self, shCoeffs, smooth = False):

        assert (isinstance(shCoeffs, torch.Tensor) and shCoeffs.dim() == 2 and shCoeffs.shape[1] == 3)

        if smooth:
            smoothed_coeffs = self.smoothSH(shCoeffs.transpose(0, 1), 4)
        else:
            smoothed_coeffs =  shCoeffs.transpose(0, 1) #self.smoothSH(shCoeffs.transpose(0, 1), 4) #smooth the first three bands?

        res = self.resolution

        theta = self.theta
        phi =  self.phi
        result = torch.zeros(res[0], res[1], smoothed_coeffs.shape[0], device=smoothed_coeffs.device)
        bands = int(math.sqrt(smoothed_coeffs.shape[1]))
        i = 0

        for l in range(bands):
            for m in range(-l, l + 1):
                sh_factor = self.SH(l, m, theta, phi)
                result = result + sh_factor.view(sh_factor.shape[0], sh_factor.shape[1], 1) * smoothed_coeffs[:, i]
                i += 1
        result = torch.max(result, torch.zeros(res[0], res[1], smoothed_coeffs.shape[0], device=smoothed_coeffs.device))
        return result
