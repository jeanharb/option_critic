import theano, sys, copy
import theano.tensor as T
import numpy as np
from collections import OrderedDict
from nnet import Model
from theano.ifelse import ifelse
from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams

class MLP3D():
  def __init__(self, num_options, model_network, temp=1):
    self.temp = temp
    self.options_W = theano.shared(np.random.uniform(
        size=(num_options, model_network[-2]["out_size"], model_network[-1]["out_size"]), high=1, low=-1))
    self.options_b = theano.shared(np.zeros((num_options, model_network[-1]["out_size"])))

    self.params = [self.options_W, self.options_b]

  def apply(self, inputs, option):
    W = self.options_W[option]
    b = self.options_b[option]

    dots = T.sum(inputs.dimshuffle(0,1,'x')*W, axis=1)
    return T.nnet.softmax((dots + b)/self.temp)

  def save_params(self):
    return [i.get_value() for i in self.params]

  def load_params(self, values):
    print "LOADING NNET..",
    for p, value in zip(self.params, values):
      p.set_value(value)
    print "LOADED"

class OptionCritic_Network():
  def __init__(self, model_network=None, gamma=0.99, learning_method="rmsprop", actor_lr=0.00025,
    batch_size=32, input_size=None, learning_params=None, dnn_type=True, clip_delta=0,
    scale=255., freeze_interval=100, grad_clip=0, termination_reg=0, num_options=8,
    double_q=False, temp=1, entropy_reg=0, BASELINE=False, **kwargs):
    x = T.ftensor4()
    next_x = T.ftensor4()
    a = T.ivector()
    o = T.ivector()
    r = T.fvector()
    terminal = T.ivector()
    self.freeze_interval = freeze_interval

    self.theano_rng = MRG_RandomStreams(1000)

    self.x_shared = theano.shared(np.zeros(tuple([batch_size]+input_size[1:]), dtype='float32'))
    self.next_x_shared = theano.shared(np.zeros(tuple([batch_size]+input_size[1:]), dtype='float32'))
    self.a_shared = theano.shared(np.zeros((batch_size), dtype='int32'))
    self.o_shared = theano.shared(np.zeros((batch_size), dtype='int32'))
    self.terminal_shared = theano.shared(np.zeros((batch_size), dtype='int32'))
    self.r_shared = theano.shared(np.zeros((batch_size), dtype='float32'))

    state_network = model_network[:-1]
    termination_network = copy.deepcopy([model_network[-1]])
    termination_network[0]["activation"] = "sigmoid"
    print "NUM OPTIONS --->", num_options
    termination_network[0]["out_size"] = num_options
    option_network = copy.deepcopy([model_network[-1]])
    option_network[0]["activation"] = "softmax"
    Q_network = copy.deepcopy([model_network[-1]])
    Q_network[0]["out_size"] = num_options

    self.state_model = Model(state_network, input_size=input_size, dnn_type=dnn_type)
    self.state_model_prime = Model(state_network, input_size=input_size, dnn_type=dnn_type)
    output_size = [None,model_network[-2]["out_size"]]
    self.Q_model = Model(Q_network, input_size=output_size, dnn_type=dnn_type)
    self.Q_model_prime = Model(Q_network, input_size=output_size, dnn_type=dnn_type)
    self.termination_model = Model(termination_network, input_size=output_size, dnn_type=dnn_type)
    self.options_model = MLP3D(num_options, model_network, temp=temp)

    s = self.state_model.apply(x/scale)
    next_s = self.state_model.apply(next_x/scale)
    next_s_prime = self.state_model_prime.apply(next_x/scale)

    termination_probs = self.termination_model.apply(theano.gradient.disconnected_grad(s))
    option_term_prob = termination_probs[T.arange(o.shape[0]), o]
    next_termination_probs = self.termination_model.apply(theano.gradient.disconnected_grad(next_s))
    next_option_term_prob = next_termination_probs[T.arange(o.shape[0]), o]
    termination_sample = T.gt(option_term_prob, self.theano_rng.uniform(size=o.shape))

    Q = self.Q_model.apply(s)
    next_Q = self.Q_model.apply(next_s)
    next_Q_prime = theano.gradient.disconnected_grad(self.Q_model_prime.apply(next_s_prime))

    disc_option_term_prob = theano.gradient.disconnected_grad(next_option_term_prob)

    action_probs = self.options_model.apply(s, o)
    sampled_actions = T.argmax(self.theano_rng.multinomial(pvals=action_probs, n=1), axis=1).astype("int32")

    if double_q:
      print "TRAINING DOUBLE_Q"
      y = r + (1-terminal)*gamma*(
        (1-disc_option_term_prob)*next_Q_prime[T.arange(o.shape[0]), o] +
        disc_option_term_prob*next_Q_prime[T.arange(next_Q.shape[0]), T.argmax(next_Q, axis=1)])
    else:
      y = r + (1-terminal)*gamma*(
        (1-disc_option_term_prob)*next_Q_prime[T.arange(o.shape[0]), o] +
        disc_option_term_prob*T.max(next_Q_prime, axis=1))

    y = theano.gradient.disconnected_grad(y)

    option_Q = Q[T.arange(o.shape[0]), o]
    td_errors = y - option_Q

    if clip_delta > 0:
      quadratic_part = T.minimum(abs(td_errors), clip_delta)
      linear_part = abs(td_errors) - quadratic_part
      td_cost = 0.5 * quadratic_part ** 2 + clip_delta * linear_part
    else:
      td_cost = 0.5 * td_errors ** 2

    #critic updates
    critic_cost = T.sum(td_cost)
    critic_params = self.Q_model.params + self.state_model.params
    learning_algo = self.Q_model.get_learning_method(learning_method, **learning_params)
    grads = T.grad(critic_cost, critic_params)
    critic_updates = learning_algo.apply(critic_params, grads, grad_clip=grad_clip)

    #actor updates
    actor_params = self.termination_model.params + self.options_model.params
    learning_algo = self.termination_model.get_learning_method("sgd", lr=actor_lr)
    disc_Q = theano.gradient.disconnected_grad(option_Q)
    disc_V = theano.gradient.disconnected_grad(T.max(Q, axis=1))
    term_grad = T.sum(option_term_prob*(disc_Q-disc_V+termination_reg))
    entropy = -T.sum(action_probs*T.log(action_probs))
    if not BASELINE:
      policy_grad = -T.sum(T.log(action_probs[T.arange(a.shape[0]), a]) * y) - entropy_reg*entropy
    else:
      policy_grad = -T.sum(T.log(action_probs[T.arange(a.shape[0]), a]) * (y-disc_Q)) - entropy_reg*entropy
    grads = T.grad(term_grad+policy_grad, actor_params)
    actor_updates = learning_algo.apply(actor_params, grads, grad_clip=grad_clip)

    if self.freeze_interval > 1:
      target_updates = OrderedDict()
      for t, b in zip(self.Q_model_prime.params+self.state_model_prime.params,
                        self.Q_model.params+self.state_model.params):
        target_updates[t] = b
      self._update_target_params = theano.function([], [], updates=target_updates)
      self.update_target_params()
      print "freeze interval:", self.freeze_interval
    else:
      print "freeze interval: None"

    critic_givens = {x:self.x_shared, o:self.o_shared, r:self.r_shared,
    terminal:self.terminal_shared, next_x:self.next_x_shared}

    actor_givens = {a:self.a_shared, r:self.r_shared,
    terminal:self.terminal_shared, o:self.o_shared, next_x:self.next_x_shared}

    print "compiling...",
    self.train_critic = theano.function([], [critic_cost], updates=critic_updates, givens=critic_givens)
    self.train_actor = theano.function([s], [], updates=actor_updates, givens=actor_givens)
    self.pred_score = theano.function([], T.max(Q, axis=1), givens={x:self.x_shared})
    self.sample_termination = theano.function([s], [termination_sample,T.argmax(Q, axis=1)], givens={o:self.o_shared})
    self.sample_options = theano.function([s], T.argmax(Q, axis=1))
    self.sample_actions = theano.function([s], sampled_actions, givens={o:self.o_shared})
    self.get_action_dist = theano.function([s, o], action_probs)
    self.get_s = theano.function([], s, givens={x:self.x_shared})
    print "complete"

  def update_target_params(self):
    if self.freeze_interval > 1:
      self._update_target_params()
    return

  def predict_move(self, s):
    return self.sample_options(s)

  def predict_termination(self, s, a):
    self.a_shared.set_value(a)
    return tuple(self.sample_termination(s))

  def get_q_vals(self, x):
    self.x_shared.set_value(x)
    return self.pred_score()[:,np.newaxis]

  def get_state(self, x):
    self.x_shared.set_value(x)
    return self.get_s()

  def get_action(self, s, o):
    self.o_shared.set_value(o)
    return self.sample_actions(s)

  def train_conv_net(self, train_set_x, next_x, options, r, terminal, actions=None, model=""):
    self.next_x_shared.set_value(next_x)
    self.o_shared.set_value(options)
    self.r_shared.set_value(r)
    self.terminal_shared.set_value(terminal)
    if model == "critic":
        self.x_shared.set_value(train_set_x)
        return self.train_critic()
    elif model == "actor":
      self.a_shared.set_value(actions)
      return self.train_actor(train_set_x)
    else:
      print "WRONG MODEL NAME"
      raise NotImplementedError

  def save_params(self):
    return [self.state_model.save_params(), self.Q_model.save_params(), self.termination_model.save_params(),
    self.options_model.save_params()]

  def load_params(self, values):
    self.state_model.load_params(values[0])
    self.Q_model.load_params(values[1])
    self.termination_model.load_params(values[2])
    self.options_model.load_params(values[3])
