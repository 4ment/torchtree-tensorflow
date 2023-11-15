import tensorflow as tf
import tensorflow_probability as tfp
import torch
from torchtree.core.utils import process_object
from torchtree.evolution.site_model import UnivariateDiscretizedSiteModel


class GammaSiteModel(UnivariateDiscretizedSiteModel):
    @property
    def shape(self) -> torch.Tensor:
        return self._parameter.tensor

    def inverse_cdf(
        self, parameter: torch.Tensor, quantile: torch.Tensor, invariant: torch.Tensor
    ) -> torch.Tensor:
        if invariant is not None:
            return torch.cat(
                (
                    torch.zeros_like(invariant),
                    GammaQuantileFunction.apply(quantile, parameter),
                ),
                dim=-1,
            )
        else:
            return GammaQuantileFunction.apply(quantile, parameter)

    @classmethod
    def from_json(cls, data, dic):
        id_ = data["id"]
        shape = process_object(data["shape"], dic)
        categories = data["categories"]
        invariant = None
        if "invariant" in data:
            invariant = process_object(data["invariant"], dic)
        if "mu" in data:
            mu = process_object(data["mu"], dic)
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
        tf_quantile = tf.constant(quantiles.numpy(), name="q")

        if shape.requires_grad:
            with tf.GradientTape(persistent=True) as tape:
                tf_shape = tf.Variable(shape.detach().numpy(), name="shape")
                dist = tfp.distributions.Gamma(concentration=tf_shape, rate=tf_shape)
                tf_rates = dist.quantile(tf_quantile)
                # gradient returns sum_i d tf_rates[i])/d tf_shape
                # but we need the individual derivatives
                grad = tf.concat(
                    [tape.gradient(tf_rate, tf_shape) for tf_rate in tf_rates], -1
                )
                ctx.shape_grad = torch.tensor(grad.numpy(), dtype=shape.dtype)
        else:
            tf_shape = tf.constant(shape.detach().numpy(), name="shape")
            dist = tfp.distributions.Gamma(concentration=tf_shape, rate=tf_shape)
            tf_rates = dist.quantile(tf_quantile)
        return torch.tensor(tf_rates.numpy(), dtype=shape.dtype, device=shape.device)

    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return None, ctx.shape_grad * grad_output
