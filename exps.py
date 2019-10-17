import jax.numpy as np

from jax import random
from jax.experimental import optimizers
from jax.api import jit, grad, vmap

import functools

import neural_tangents as nt
from neural_tangents import stax

from layers import Dense

def loss_fn(predict_fn, ys, t):
    mean, var = predict_fn(t)
    mean = np.reshape(mean, (-1,))
    var = np.diag(var)
    ys = np.reshape(ys, (-1,))

    mean_predictions = 0.5 * np.mean(ys ** 2 - 2 * mean * ys + var + mean ** 2)

    return mean_predictions

key = random.PRNGKey(10)

train_points = 5
test_points = 50
noise_scale = 1e-1

target_fn = lambda x: np.sin(x)

key, x_key, y_key = random.split(key, 3)

train_xs = random.uniform(x_key, (train_points, 1), minval=-np.pi, maxval=np.pi)

train_ys = target_fn(train_xs)
train_ys += noise_scale * random.normal(y_key, (train_points, 1))
train = (train_xs, train_ys)

test_xs = np.linspace(-np.pi, np.pi, test_points)
test_xs = np.reshape(test_xs, (test_points, 1))

test_ys = target_fn(test_xs)
test = (test_xs, test_ys)

custom = True
if custom == False:
    init_fn, apply_fn, kernel_fn = stax.serial(
        stax.Dense(10, W_std=1., b_std=0.0), stax.Erf(),
        stax.Dense(10, W_std=1., b_std=0.0), stax.Erf(),
        stax.Dense(1, W_std=1., b_std=0.0)
    )
else:
    init_fn, apply_fn, kernel_fn = stax.serial(
        Dense(5, W_std=1., b_std=0.0), stax.Erf(),
        Dense(5, W_std=1., b_std=0.0), stax.Erf(),
        Dense(1, W_std=1., b_std=0.0)
    )


apply_fn = jit(apply_fn)
kernel_fn = jit(kernel_fn, static_argnums=(2,))


learning_rate = 0.1
training_steps = 10000

opt_init, opt_update, get_params = optimizers.sgd(learning_rate)
opt_update = jit(opt_update)


# regularize??
loss = jit(lambda params, x, y: 0.5 * np.mean((apply_fn(params, x) - y) ** 2) )
grad_loss = jit(lambda state, x, y: grad(loss)(get_params(state), x, y))

train_losses = []
test_losses = []

key, net_key = random.split(key)
_, params = init_fn(net_key, (-1, 1))

print(params[0])


opt_state = opt_init(params)

for i in range(training_steps):
  opt_state = opt_update(i, grad_loss(opt_state, *train), opt_state)

  train_losses += [loss(get_params(opt_state), *train)]  
  test_losses += [loss(get_params(opt_state), *test)]

  if i % 1000 == 1:

        print(loss(get_params(opt_state), *train))
        print(loss(get_params(opt_state), *test))

print('NTK')

train_predict_fn = nt.predict.gradient_descent_mse_gp(
    kernel_fn, train_xs, train_ys, train_xs, 'ntk', 1e-4, compute_var=True)
test_predict_fn = nt.predict.gradient_descent_mse_gp(
    kernel_fn, train_xs, train_ys, test_xs, 'ntk', 1e-4, compute_var=True)

train_loss_fn = functools.partial(loss_fn, train_predict_fn, train_ys)
test_loss_fn = functools.partial(loss_fn, test_predict_fn, test_ys)

ts = np.arange(0, 10 ** 3, 10 ** -1)
ntk_train_loss_mean = vmap(train_loss_fn)(ts)
ntk_test_loss_mean = vmap(test_loss_fn)(ts)

print(train_losses[-1])
print(ntk_train_loss_mean[-1])
print(test_losses[-1])
print(ntk_test_loss_mean[-1])

