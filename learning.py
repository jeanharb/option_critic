import theano, itertools
import numpy as np
from collections import OrderedDict
from theano import tensor

def get_learning_method(l_method, **kwargs):
    if l_method == "adam":
      return Adam(**kwargs)
    elif l_method == "adadelta":
      return AdaDelta(**kwargs)
    elif l_method == "sgd":
      return SGD(**kwargs)
    elif l_method == "rmsprop":
      return RMSProp(**kwargs)

class SGD():
  def __init__(self, lr=0.01):
    self.lr = lr

  def apply(self, params, grads, grad_clip=0):
    updates = OrderedDict()
    for param, grad in zip(params, grads):
      if grad_clip > 0:
          grad = tensor.clip(grad, -grad_clip, grad_clip)
      updates[param] = param - self.lr * grad
    return updates

class RMSProp():
  def __init__(self, rho=0.95, eps=1e-4, lr=0.001):
    self.rho = rho
    self.eps = eps
    self.lr = lr
    print "rms", rho, eps, lr

  def apply(self, params, grads, grad_clip=0):
    updates = OrderedDict()

    for param, grad in zip(params, grads):
        acc_grad = theano.shared(param.get_value() * 0)
        acc_grad_new = self.rho * acc_grad + (1 - self.rho) * grad

        acc_rms = theano.shared(param.get_value() * 0)
        acc_rms_new = self.rho * acc_rms + (1 - self.rho) * grad ** 2

        updates[acc_grad] = acc_grad_new
        updates[acc_rms] = acc_rms_new

        updates[param] = (param - self.lr * 
                          (grad / 
                           tensor.sqrt(acc_rms_new - acc_grad_new ** 2 + self.eps)))

    return updates

class Adam():
  def __init__(self, lr=0.0005, beta1=0.9, beta2=0.999, epsilon=1e-4):
    self.lr = lr
    self.b1 = beta1
    self.b2 = beta2
    self.eps = epsilon

  def apply(self, params, grads):
    t = theano.shared(np.array(2., dtype='float32'))
    updates = OrderedDict()
    updates[t] = t+1
    for param, grad in zip(params, grads):
      last_1_moment = theano.shared(param.get_value() * 0)
      last_2_moment = theano.shared(param.get_value() * 0)
      new_last_1_moment = (1 - self.b1) * grad + self.b1 * last_1_moment
      new_last_2_moment = (1 - self.b2) * grad**2 + self.b2 * last_2_moment

      updates[last_1_moment] = new_last_1_moment
      updates[last_2_moment] = new_last_2_moment
      updates[param] = (param - (self.lr*(new_last_1_moment/(1-self.b1**t)) /
                  (tensor.sqrt(new_last_2_moment/(1-self.b2**t)) + self.eps)))#z.astype("float32")
    return updates

class AdaDelta():
  def __init__(self, rho=0.95, rho2=0.95):
    self.rho = rho
    self.rho2 = rho2

  def apply(self, params, grads):
    zipped_grads = [theano.shared(p.get_value() * 0) for p in params]
    running_up2 = [theano.shared(p.get_value() * 0) for p in params]
    running_grads2 = [theano.shared(p.get_value() * 0) for p in params]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, self.rho * rg2 + (1-self.rho) * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    #f_grad_shared = theano.function(input_params, cost, updates=zgup + rg2up,
    #                                name='adadelta_f_grad_shared')

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, self.rho2 * ru2 + (1-self.rho2) * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(params, updir)]

    #f_update = theano.function([], [], updates=ru2up + param_up, name='adadelta_f_update')
    updates = ru2up + param_up

    return updates


