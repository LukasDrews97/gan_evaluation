# MIT License
#
# Copyright (c) 2021 Min Jin Chong
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# This class is partially derived from these two files:
# https://github.com/PyTorchLightning/metrics/blob/master/torchmetrics/image/inception.py
# https://github.com/mchong6/FID_IS_infinity/blob/master/score_infinity.py

import torch
from torch import Tensor
from torchmetrics import Metric
import numpy as np
from sklearn.linear_model import LinearRegression
from torchmetrics.image.fid import _compute_fid
from torchmetrics.utilities.data import dim_zero_cat
from .inception import load_inception_net
import torch.nn.functional as F
from typing import Any, Callable, List, Optional, Tuple
from torchmetrics.utilities import rank_zero_warn

class FIDInfinity(Metric):
    r"""
    Calculates Unbiased Fréchet inception distance which is used to access the quality of generated images.
    Args:
        num_im:
            Number of images to calculate metric on. Must be at least 5000.
        num_points:
            Number of points to fit linear regression.
        compute_on_step:
            Forward only calls ``update()`` and return ``None`` if this is set to ``False``.
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step
        process_group:
            Specify the process group on which synchronization is called.
            default: ``None`` (which selects the entire world)
        dist_sync_fn:
            Callback that performs the allgather operation on the metric state. When ``None``, DDP
            will be used to perform the allgather
    References:
        [1] Effectively Unbiased FID and Inception Score and where to find them
        Min Jin Chong, David Forsyth
        https://arxiv.org/abs/1911.07023
        [2] Rethinking the Inception Architecture for Computer Vision
        Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna
        https://arxiv.org/abs/1512.00567
        [3] GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium,
        Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, Sepp Hochreiter
        https://arxiv.org/abs/1706.08500
    Raises:
        AssertionError:
            If num_im is less than 5000.
    """
    real_features: List[Tensor]
    fake_features: List[Tensor]

    def __init__(
        self,
        num_im: int = 50000,
        num_points: int = 15,
        compute_on_step: bool = False,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable[[Tensor], List[Tensor]] = None,
    ) -> None:
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.num_im = num_im
        self.num_points = num_points
        self.inception = load_inception_net()

        rank_zero_warn(
            "Metric `FIDInfinity` will save all extracted features in buffer."
            " For large datasets this may lead to large memory footprint.",
            UserWarning,
        )

        assert num_im >= 5000

        self.add_state("real_features", [], dist_reduce_fx=None)
        self.add_state("fake_features", [], dist_reduce_fx=None)  


    def update(self, imgs: Tensor, real: bool) -> None:
        """Update the state with extracted features.
        Args:
            imgs: 
                Tensor with images feed to the feature extractor
            real: 
                Bool indicating if images belong to the real or the fake distribution
        """
        features, _ = self.accumulate_activations(imgs)
        if real:
            self.real_features.append(features)
        else:
            self.fake_features.append(features)

    # This method is partially derived from https://github.com/mchong6/FID_IS_infinity/blob/master/score_infinity.py
    def compute(self) -> np.float64:
        """Calculate Unbiased FID score based on accumulated extracted features from the two distributions."""
        assert len(dim_zero_cat(self.real_features)) >= 5000
        assert len(dim_zero_cat(self.fake_features)) >= 5000
        
        fids = []

        # Choose the number of images to evaluate FID_N at regular intervals over N
        fid_batches = np.linspace(5000, self.num_im, self.num_points).astype('int32')
        

        # Evaluate FID_N
        for fid_batch_size in fid_batches:
            # sample with replacement

            real = dim_zero_cat(self.real_features)
            fake = dim_zero_cat(self.fake_features)
            #np.random.shuffle(real)
            np.random.shuffle(fake)

            fids.append(self.calculate_FID(real[:fid_batch_size], fake[:fid_batch_size]))

        fids = np.array(fids).reshape(-1, 1)
        
        # Fit linear regression
        reg = LinearRegression().fit(1/fid_batches.reshape(-1, 1), fids)
        fid_infinity = reg.predict(np.array([[0]]))[0,0]

        return fid_infinity
        
    def get_number_of_features(self):
        """Return the total number of stored features"""
        len_real = len(dim_zero_cat(self.real_features))
        len_fake = len(dim_zero_cat(self.fake_features))
        assert len_real == len_fake
        return len_real

    # This method is borrowed from https://github.com/PyTorchLightning/metrics/blob/master/torchmetrics/image/fid.py
    def calculate_FID(self, real_features: List[Tensor], fake_features: List[Tensor]) -> Tensor:
        """Calculate FID score based on accumulated extracted features from the two distributions."""
        real_features = dim_zero_cat(real_features)
        fake_features = dim_zero_cat(fake_features)
        # computation is extremely sensitive so it needs to happen in double precision
        orig_dtype = real_features.dtype
        real_features = real_features.double()
        fake_features = fake_features.double()

        # calculate mean and covariance
        n = real_features.shape[0]
        mean1 = real_features.mean(dim=0)
        mean2 = fake_features.mean(dim=0)
        diff1 = real_features - mean1
        diff2 = fake_features - mean2
        cov1 = 1.0 / (n - 1) * diff1.t().mm(diff1)
        cov2 = 1.0 / (n - 1) * diff2.t().mm(diff2)

        # compute fid
        return _compute_fid(mean1, cov1, mean2, cov2).to(orig_dtype)

    # This method is borrowed from https://github.com/mchong6/FID_IS_infinity/blob/master/score_infinity.py
    def accumulate_activations(self, images: Tensor) -> Tuple[Tensor, Tensor]:
        """Accumulate activations through inceptionv3"""
        pool, logits = [], []
        pool_val, logits_val = self.inception(images)
        pool += [pool_val]
        logits += [F.softmax(logits_val, 1)]
        
        pool =  torch.cat(pool, 0)[:self.num_im]
        logits = torch.cat(logits, 0)[:self.num_im]

        return pool, logits
