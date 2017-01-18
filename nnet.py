import theano, lasagne
import theano.tensor as T
import math, csv, time, sys, os, copy
from collections import OrderedDict
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from learning import get_learning_method
from lasagne.layers import Conv2DLayer
if theano.config.device.startswith("gpu"):
  from lasagne.layers import cuda_convnet
import numpy as np

class AddBias():
  def __init__(self, init_val=None):
    self.b = init_val if init_val is not None else theano.shared(np.array([0.1], dtype="float32")[0])

  def get_output_for(self, inputs, deterministic=None):
    return inputs+self.b

  def get_params(self):
    return [self.b]

class Model():
  def get_activation(self, activation):
    if activation == "softmax":
      output = T.nnet.softmax
    elif activation == "tanh":
      output = T.tanh
    elif activation == "relu":
      output = T.nnet.relu
    elif activation == "linear":
      output = None
    elif activation == "sigmoid":
      output = T.nnet.sigmoid
    elif activation == "hard_sigmoid":
      output = T.nnet.hard_sigmoid
    return output

  def create_layer(self, inputs, model, dnn_type=True):
    inits = {"zeros": lasagne.init.Constant(0.)}
    if model["model_type"] == "conv":
      if dnn_type:
        import lasagne.layers.dnn as dnn
      conv_type = dnn.Conv2DDNNLayer if dnn_type else Conv2DLayer#cuda_convnet.Conv2DCCLayer
      poolsize = tuple(model["pool"]) if "pool" in model else (1,1)
      stride = tuple(model["stride"]) if "stride" in model else (1,1)
      layer = conv_type(inputs, 
        model["out_size"], 
        filter_size=model["filter_size"], 
        stride=stride, 
        nonlinearity=self.get_activation(model["activation"]),
        W=lasagne.init.GlorotUniform() if "W" not in model else inits[model["W"]],
        b=lasagne.init.Constant(.1) if "b" not in model else inits[model["b"]])
    elif model["model_type"] == "mlp" or model["model_type"] == "logistic":
      layer = lasagne.layers.DenseLayer(inputs,
        num_units=model["out_size"],
        nonlinearity=self.get_activation(model["activation"]),
        W=lasagne.init.GlorotUniform() if "W" not in model else (inits[model["W"]] if isinstance(model["W"], basestring) else lasagne.init.Constant(model["W"])),
        b=lasagne.init.Constant(.1) if "b" not in model else (inits[model["b"]] if isinstance(model["b"], basestring) else lasagne.init.Constant(model["b"])))
    elif model["model_type"] == "bias":
      layer = AddBias(model["b"] if "b" in model else None)
    else:
      print "UNKNOWN LAYER NAME"
      raise NotImplementedError
    return layer

  def __init__(self, model, input_size=None, rng=1234, dnn_type=True):
    """
    example model:
    model = [{"model_type": "conv", "filter_size": [5,5], "pool": [1,1], "stride": [1,1], "out_size": 5},
             {"model_type": "conv", "filter_size": [7,7], "pool": [1,1], "stride": [1,1], "out_size": 15},
             {"model_type": "mlp", "out_size": 300, "activation": "tanh"},
             {"model_type": "classification", "out_size": 10, "activation": "tanh", "loss_type": "log_loss"}]
    """
    rng = np.random.RandomState(rng)
    self.theano_rng = RandomStreams(rng.randint(2 ** 30))
    lasagne.random.set_rng(rng) #set rng

    new_layer = tuple(input_size) if isinstance(input_size, list) else input_size
    print model
    self.model = model
    self.input_size = input_size
    self.dnn_type = dnn_type

    # create neural net layers
    self.params = []
    self.layers = []
    for i, m in enumerate(model):
      new_layer = self.create_layer(new_layer, m, dnn_type=dnn_type)
      self.params += new_layer.get_params()
      self.layers.append(new_layer)

  def apply(self, x, deterministic=False):
    last_layer_inputs = x
    last_model_type = None
    for i, m in enumerate(self.model):
      if m["model_type"] in ["mlp", "logistic"] and last_layer_inputs.ndim > 2:
        last_layer_inputs = last_layer_inputs.flatten(2)
      last_layer_inputs = self.layers[i].get_output_for(last_layer_inputs, deterministic=deterministic)
      last_model_type = m["model_type"]
    return last_layer_inputs

  def get_learning_method(self, l_method, **kwargs):
    return get_learning_method(l_method, **kwargs)

  def save_params(self):
    return [i.get_value() for i in self.params]

  def load_params(self, values):
    print "LOADING NNET..",
    for p, value in zip(self.params, values):
      p.set_value(value)
    print "LOADED"
