import tensorflow as tf
import tensorflow_probability as tfp
import torch
from torchtree import Parameter

from torchflow.site_model import GammaQuantileFunction, GammaSiteModel


def test_simple():
    shape = torch.tensor([1.0], requires_grad=True)
    parameter = Parameter('p', shape)
    site_model = GammaSiteModel('id', parameter, 4)
    expected = torch.tensor([0.1457844, 0.5131316, 1.0708310, 2.2702530])
    assert torch.allclose(site_model.rates(), expected)


def test_batch():
    expected = torch.tensor(
        [
            [0.1457844, 0.5131316, 1.0708310, 2.2702530],
            [1.166722e-08, 6.889589e-04, 1.145135e-01, 3.884798],
        ]
    )
    parameter = torch.tensor([[1.0], [0.1]])
    site_model = GammaSiteModel('id', Parameter('p', parameter), 4)
    assert torch.allclose(site_model.rates(), expected)


def test_quantile():
    shape = torch.tensor([0.10], requires_grad=True)
    quantile = torch.tensor([0.2])
    rates = GammaQuantileFunction.apply(quantile, shape)
    assert torch.allclose(torch.tensor([6.218802e-07]), rates)
    rates.backward()
    # eps=1e-07
    # (qgamma(0.2, 0.1+eps, 0.1+eps) - qgamma(0.2, 0.1-eps, 0.1-eps))/(eps*2)
    assert torch.allclose(torch.tensor([9.433518e-05]), shape.grad)


def test_quantiles():
    quantiles = torch.tensor([0.2, 0.5])
    shape = torch.tensor([0.10], requires_grad=True)
    tf_quantiles = tf.constant(quantiles.numpy(), name='q')

    with tf.GradientTape() as tape:
        tf_shape = tf.Variable(shape.detach().numpy(), name='shape')
        dist = tfp.distributions.Gamma(concentration=tf_shape, rate=tf_shape)
        tf_rates = dist.quantile(tf_quantiles)
        tf_sum = tf.math.reduce_sum(tf_rates * tf_quantiles)
        tf_grad = tape.gradient(tf_sum, tf_shape)

    rates = GammaQuantileFunction.apply(quantiles, shape)
    sum = torch.sum(quantiles * rates)
    sum.backward()

    assert torch.allclose(torch.tensor(tf_grad.numpy()), shape.grad)
