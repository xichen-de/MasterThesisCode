#  MIT License
#
#  Copyright 2021 Xi Chen
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy as np

from stable_baselines.common.noise import ActionNoise


class LinearNormalActionNoise(ActionNoise):
    """
    A gaussian action noise with linear decay for the standard deviation.

    :param mean: (np.ndarray) the mean value of the noise
    :param sigma: (np.ndarray) the scale of the noise (std here)
    :param max_steps: (int)
    :param final_sigma: (np.ndarray)
    """

    def __init__(self, mean, sigma, max_steps, final_sigma=None):
        self._mu = mean
        self._sigma = sigma
        self._step = 0
        self._max_steps = max_steps
        if final_sigma is None:
            final_sigma = np.zeros_like(sigma)
        self._final_sigma = final_sigma

    def __call__(self):
        t = min(1.0, self._step / self._max_steps)
        sigma = (1 - t) * self._sigma + t * self._final_sigma
        self._step += 1
        return np.random.normal(self._mu, sigma)
