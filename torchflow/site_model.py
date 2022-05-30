from typing import Optional, Union

import tensorflow as tf
import tensorflow_probability as tfp
import torch
from torchtree.core.abstractparameter import AbstractParameter
from torchtree.core.utils import process_object
from torchtree.evolution.site_model import SiteModel


class GammaSiteModel(SiteModel):
    def __init__(
        self,
        id_: Union[str, None],
        shape: AbstractParameter,
        categories: int,
        invariant: AbstractParameter = None,
        mu: AbstractParameter = None,
    ) -> None:
        super().__init__(id_, mu)
        self._shape = shape
        self.categories = categories
        self._invariant = invariant
        self.probs = torch.full(
            (categories,), 1.0 / categories, dtype=self.shape.dtype, device=shape.device
        )
        self._rates = None
        self.need_update = True

    @property
    def shape(self) -> torch.Tensor:
        return self._shape.tensor

    @property
    def invariant(self) -> torch.Tensor:
        return self._invariant.tensor if self._invariant else None

    def update_rates(self, shape: torch.Tensor, invariant: torch.Tensor):
        if invariant:
            cat = self.categories - 1
            quantile = (2.0 * torch.arange(cat, device=shape.device) + 1.0) / (
                2.0 * cat
            )
            self.probs = torch.cat(
                (
                    invariant,
                    torch.full((cat,), (1.0 - invariant) / cat, device=shape.device),
                )
            )
            rates = torch.cat(
                (
                    torch.zeros_like(invariant),
                    GammaQuantileFunction.apply(quantile, shape),
                )
            )
        else:
            quantile = (
                2.0 * torch.arange(self.categories, device=shape.device) + 1.0
            ) / (2.0 * self.categories)
            rates = GammaQuantileFunction.apply(quantile, shape)

        self._rates = rates / (rates * self.probs).sum(-1, keepdim=True)
        if self._mu is not None:
            self._rates *= self._mu.tensor

    def rates(self) -> torch.Tensor:
        if self.need_update:
            self.update_rates(self.shape, self.invariant)
            self.need_update = False
        return self._rates

    def probabilities(self) -> torch.Tensor:
        if self.need_update:
            self.update_rates(self.shape, self.invariant)
            self.need_update = False
        return self.probs

    def handle_model_changed(self, model, obj, index):
        pass

    def handle_parameter_changed(self, variable, index, event):
        self.need_update = True
        self.fire_model_changed()

    @property
    def sample_shape(self) -> torch.Size:
        return max(
            [parameter.shape[:-1] for parameter in self._parameters.values()],
            key=len,
        )

    def cuda(self, device: Optional[Union[int, torch.device]] = None):
        super().cuda()
        self.need_update = True

    def cpu(self) -> None:
        super().cpu()
        self.need_update = True

    @classmethod
    def from_json(cls, data, dic):
        id_ = data['id']
        shape = process_object(data['shape'], dic)
        categories = data['categories']
        invariant = None
        if 'invariant' in data:
            invariant = process_object(data['invariant'], dic)
        if 'mu' in data:
            mu = process_object(data['mu'], dic)
        else:
            mu = None
        return cls(id_, shape, categories, invariant, mu)


class GammaQuantileFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        quantiles: torch.Tensor,
        shape: torch.Tensor,
    ) -> torch.Tensor:
        ctx.quantiles = quantiles
        q = tf.constant(quantiles.numpy(), name='q')

        if shape.requires_grad:
            with tf.GradientTape() as tape:
                tf_shape = tf.Variable(shape.detach().numpy(), name='shape')
                dist = tfp.distributions.Gamma(concentration=tf_shape, rate=tf_shape)
                tf_rates = dist.quantile(q)
                grad = tape.gradient(tf_rates, tf_shape)
                ctx.shape_grad = torch.tensor(grad.numpy(), dtype=shape.dtype)
        else:
            tf_shape = tf.constant(shape.detach().numpy(), name='shape')
            dist = tfp.distributions.Gamma(concentration=tf_shape, rate=tf_shape)
            tf_rates = dist.quantile(q)
        return torch.tensor(tf_rates.numpy(), dtype=shape.dtype, device=shape.device)

    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return None, torch.sum(ctx.shape_grad * grad_output * ctx.quantiles, -1)
