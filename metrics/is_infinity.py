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
from typing import Any, Callable, List, Optional, Tuple
from torchmetrics import Metric
from sklearn.linear_model import LinearRegression
from .inception import load_inception_net
from torchmetrics.utilities.data import dim_zero_cat
import numpy as np
import torch.nn.functional as F
from torchmetrics.utilities import rank_zero_warn

class ISInfinity(Metric):
    r"""
    Calculates Unbiased Inception score which is used to access the quality of generated images.
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
        [2] Improved Techniques for Training GANs
        Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, Xi Chen
        https://arxiv.org/abs/1606.03498
        [3] GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium,
        Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, Sepp Hochreiter
        https://arxiv.org/abs/1706.08500
    Raises:
        AssertionError:
            If num_im is less than 5000.
    """
    features: List[Tensor]

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

        self.num_points = num_points
        self.num_im = num_im
        self.add_state("features", [], dist_reduce_fx=None)
        self.inception = load_inception_net()

        assert num_im >= 5000

        rank_zero_warn(
            "Metric `ISInfinity` will save all extracted features in buffer."
            " For large datasets this may lead to large memory footprint.",
            UserWarning,
        )

    def update(self, imgs: Tensor) -> None:
        """Update the state with extracted features.
        Args:
            imgs: 
                Tensor with images feed to the feature extractor
        """
        _, logits = self.accumulate_activations(imgs)
        self.features.append(logits)

    # This method is partially derived from https://github.com/mchong6/FID_IS_infinity/blob/master/score_infinity.py
    def compute(self) -> np.float64:
        """Calculate effectively unbiased IS_inf using extrapolation."""
        assert len(dim_zero_cat(self.features)) >= 5000

        IS = []

        # Choose the number of images to evaluate IS_N at regular intervals over N
        IS_batches = np.linspace(5000, self.num_im, self.num_points).astype('int32')

        # Evaluate IS_N
        for IS_batch_size in IS_batches:
            # sample with replacement
            IS_logits = dim_zero_cat(self.features)
            np.random.shuffle(IS_logits)
            is_score = self.calculate_inception_score(IS_logits[:IS_batch_size].cpu().numpy())[0]
            IS.append(is_score)

        IS = np.array(IS).reshape(-1, 1)

        # Fit linear regression
        reg = LinearRegression().fit(1/IS_batches.reshape(-1, 1), IS)
        IS_infinity = reg.predict(np.array([[0]]))[0,0]
        return IS_infinity

    def get_number_of_features(self):
        """Return the total number of stored features"""
        return len(dim_zero_cat(self.features))

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
        
    # This method is borrowed from https://github.com/mchong6/FID_IS_infinity/blob/master/score_infinity.py
    def calculate_inception_score(self, pred: np.ndarray, num_splits: int = 1) -> Tuple[np.float32, np.float32]:
        """Calculate Inception score"""
        scores = []
        for index in range(num_splits):
            pred_chunk = pred[index * (pred.shape[0] // num_splits): (index + 1) * (pred.shape[0] // num_splits), :]
            kl_inception = pred_chunk * (np.log(pred_chunk) - np.log(np.expand_dims(np.mean(pred_chunk, 0), 0)))
            kl_inception = np.mean(np.sum(kl_inception, 1))
            scores.append(np.exp(kl_inception))

        return np.mean(scores), np.std(scores)
