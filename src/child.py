# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""AWD ENAS fixed model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import data_utils
import utils


flags = tf.app.flags
FLAGS = flags.FLAGS


flags.DEFINE_integer('child_batch_size', 16, '')
flags.DEFINE_integer('child_bptt_steps', 70, '')

global emb
def _lstm(sample_arc, x,prev_c, prev_h,w_0, w_lstm,
            params):
  """Multi-layer LSTM.
  Args:
    sample_arc: [5 * num_layers], sequence of tokens representing architecture.
    x: [batch_size, num_steps, input_size].
    prev_h:  [batch_size, hidden_size].
    prev_c: [batch_size, hidden_size].
    w_0 : [emb_size , hid_size]
    w_lstm: [None, [inp + hid * hidden_size] * (num_layers)].
    params: hyper-params object.
  Returns:
    next_h: [batch_size, hidden_size].
    next_c: [batch_size, hidden_size].
    all_h: [batch_size, num_steps, hidden_size].
  """
  batch_size = x.get_shape()[0].value
  num_steps = tf.shape(x)[1]
  num_layers = len(sample_arc) // 5
  print("num of layers= {}".format(num_layers))
  emb = x
  all_h = tf.TensorArray(dtype=tf.float32, size=num_steps, infer_shape=False)

  w_real = [] #adding the w_lstm for chosen conns, fns, and op
  start_idx = 5

  for layer_id in range(num_layers - 2):
    prev_idx_1 = sample_arc[start_idx]
    prev_idx_2 = sample_arc[start_idx + 2]
    func_idx_1 = sample_arc[start_idx + 1]
    func_idx_2 = sample_arc[start_idx + 3]
    op_id = sample_arc[start_idx + 4]
    w_real.append(w_lstm[layer_id][prev_idx_1, prev_idx_2,
                  func_idx_1, func_idx_2, op_id])
    start_idx += 5

  w_lstm = w_real
  vars_h = [w_0] + w_lstm


  def _select_function(h, function_id):
    h = tf.stack([tf.tanh(h), tf.sigmoid(h), h, h * 0], axis=0)
    h = h[function_id]
    return h

  def _select_op(h_1, h_2, op_id):
    h = tf.stack([tf.add(h_1, h_2), tf.multiply(h_1, h_2)], axis= 0)
    h = h[op_id]
    return h

  def _condition(step, *unused_args):
    return tf.less(step, num_steps)

  def _body(step, pprev_c, pprev_h, all_h):
    """Body function."""
    inp = emb[:, step, :]
    print("shape of inp is: {}".format(inp.get_shape()))

    layers = []
    used_h1 = []
    used_h2 = []
    hid_size = params.hidden_size
    print("entered the body")

    p_h = pprev_h
    p_c = pprev_c

    start_idx = 5
    for layer_id in range(num_layers):
      inp = emb[:, step, :]

      #inp_size = tf.shape(inp)[-1] if layer_id < 4 else tf.shape(p_h)[-1]
      print("layer_id is {}".format(layer_id))

      if layer_id < 3:
        next_c = p_c

      elif layer_id == 3 : next_c = layers[2]

      if layer_id == 0:
        h = tf.matmul(p_h, (tf.split(w_0, [tf.shape(inp)[-1], tf.shape(p_h)[-1]], axis=0)[1]))
        x = tf.matmul(inp, (tf.split(w_0, [tf.shape(inp)[-1], tf.shape(p_h)[-1]], axis=0)[0]))
        h = tf.tanh(h)
        x = tf.sigmoid(x)
        ht = x + h
        ht.set_shape([batch_size, hid_size])
        print("for layer_id : 0 , w_lstm has size {}".format(w_0))
        layers.append(ht)
      elif layer_id == 1:
        layers.append(p_c)
      else:
        #arc_seq = [con_1, func_1, con_2, func_2, op1 ,......,op_num_of_layers]
        prev_idx_1 = sample_arc[start_idx]
        prev_idx_2 = sample_arc[start_idx + 2]
        func_idx_1 = sample_arc[start_idx + 1]
        func_idx_2 = sample_arc[start_idx + 3]
        op_id = sample_arc[start_idx + 4]
        print("for layer_id : {} , w_lstm has size {}".format(layer_id, w_lstm[layer_id - 2]))

        #select a previous layer to connect
        used_h1.append(tf.one_hot(prev_idx_1, depth=num_layers, dtype=tf.int32))
        used_h2.append(tf.one_hot(prev_idx_2, depth=num_layers, dtype=tf.int32))
        prev_h_1 = tf.stack(layers, axis=0)[prev_idx_1]
        prev_h_2 = tf.stack(layers, axis=0)[prev_idx_2]

        #Weighting, activation function, and selecting an operation for the connections
        h_1 = tf.matmul(prev_h_1, tf.split(w_lstm[layer_id - 2], 2, axis=0)[0])
        h_2 = tf.matmul(prev_h_2, tf.split(w_lstm[layer_id - 2], 2, axis=0)[1])
        h_1 = _select_function(h_1, func_idx_1)
        h_2 = _select_function(h_2, func_idx_2)
        h = _select_op(h_1, h_2, op_id)
        h.set_shape([batch_size, params.hidden_size])
        layers.append(h)
        start_idx += 5

    next_h = layers[-1]
    all_h = all_h.write(step, next_h)
    return step + 1, next_c, next_h, all_h

  loop_inps = [tf.constant(0, dtype=tf.int32), prev_c, prev_h, all_h]
  _, next_c, next_h, all_h = tf.while_loop(_condition, _body, loop_inps)

  all_h = tf.transpose(all_h.stack(), [1, 0, 2])

  return next_c, next_h, all_h, vars_h


def _set_default_params(params):
  """Set default hyper-parameters."""
  params.add_hparam('alpha', 0.0)  # activation L2 reg
  params.add_hparam('beta', 0.0)  # activation slowness reg
  params.add_hparam('best_valid_ppl_threshold', 5)

  params.add_hparam('batch_size', FLAGS.child_batch_size)
  params.add_hparam('bptt_steps', FLAGS.child_bptt_steps)

  # for dropouts: dropping rate, NOT keeping rate
  params.add_hparam('drop_e', 0.0)  # word
  params.add_hparam('drop_i', 0.0)  # embeddings
  params.add_hparam('drop_x', 0.0)  # input to RNN cells
  params.add_hparam('drop_l', 0.0)  # between layers
  params.add_hparam('drop_o', 0.0)  # output
  params.add_hparam('drop_w', 0.00)  # weight

  params.add_hparam('grad_bound', 0.25)
  params.add_hparam('hidden_size', 64)
  params.add_hparam('init_range', 0.04)
  params.add_hparam('learning_rate', 10.01)
  params.add_hparam('num_train_epochs', 500)
  params.add_hparam('vocab_size', 10000)
  params.add_hparam('emb_size', 64)

  params.add_hparam('weight_decay', 8e-7)
  params.add_hparam('num_layers', 4)

  return params


class LM(object):
  """Language model."""

  def __init__(self, params, controller, x_train, x_valid, name='child'):
    print('-' * 80)
    print('Building LM')

    self.params = _set_default_params(params)
    self.controller = controller
    self.sample_arc = tf.unstack(controller.sample_arc)
    self.name = name

    # train data
    (self.x_train, self.y_train,
     self.num_train_batches, self.reset_start_idx,
     self.should_reset, self.base_bptt) = data_utils.input_producer(
         x_train, params.batch_size, params.bptt_steps, random_len=True)
    params.add_hparam(
        'num_train_steps', self.num_train_batches * params.num_train_epochs)

    print("X_train size = {}".format(np.shape(x_train)))

    # valid data
    (self.x_valid, self.y_valid,
     self.num_valid_batches) = data_utils.input_producer(
         x_valid, params.batch_size, params.bptt_steps)

    self._build_params()
    self._build_train()
    self._build_valid()

  def _build_params(self):
    """Create model parameters."""

    print('-' * 80)
    print('Building model params')
    initializer = tf.initializers.random_uniform(minval=-self.params.init_range,
                                                 maxval=self.params.init_range)
    num_functions = self.params.controller_num_functions
    num_ops = self.params.controller_num_operations
    num_layers = self.params.controller_num_layers
    #hidden_size = self.params.hidden_size
    batch_size = self.params.batch_size
    #inp_size = self.params.emb_size

    with tf.variable_scope(self.name, initializer=initializer):

      with tf.variable_scope('embedding'):
        w_emb = tf.get_variable('w', [self.params.vocab_size, self.params.emb_size])
        w_soft = tf.get_variable('w_soft', [self.params.hidden_size, self.params.vocab_size])
        b = tf.get_variable('b',shape = [self.params.batch_size, self.params.vocab_size])

      with tf.variable_scope('LSTM_cell'):
        w_0 = tf.get_variable('w_0', [self.params.emb_size + self.params.hidden_size, self.params.hidden_size])
        w_lstm = []
        for layer_id in range(2, num_layers):
          #inp_size = self.params.emb_size if layer_id < 1 else self.params.hidden_size
          hidden_size= self.params.hidden_size
          with tf.variable_scope('layer_{}'.format(layer_id)):
            w = tf.get_variable(
              'w', [layer_id, layer_id, num_functions,  num_functions, num_ops,
                     2 * hidden_size, hidden_size], initializer=initializer)
            w_lstm.append(w)

      with tf.variable_scope('init_states'):
        batch_prev_c, batch_prev_h, batch_reset = [], [], []
        hidden_size = self.params.hidden_size
        init_shape = [self.params.batch_size, hidden_size]
        #  with tf.variable_scope('layer_{}'.format(layer_id)):
        batch_prev_h = tf.get_variable('h', init_shape, dtype=tf.float32, trainable=False)
        zeros = np.zeros(init_shape, dtype=np.float32)
        # batch_reset.append(tf.assign(batch_prev_c[-1], zeros))
        batch_reset.append(tf.assign(batch_prev_h, zeros))
        zeros = np.zeros(init_shape, dtype=np.float32)
        batch_prev_c = tf.get_variable('c', init_shape, dtype=tf.float32, trainable=False)
        batch_reset.append(tf.assign(batch_prev_c, zeros))
    self.num_params = sum([np.prod(v.shape) for v in tf.trainable_variables()])
    print('All children have {0} params'.format(self.num_params))

    num_params_per_child = 0
    for v in tf.trainable_variables():
      if v.name.startswith(self.name):
        if 'LSTM_cell' in v.name:
          num_params_per_child += v.shape[-2].value * v.shape[-1].value
        else:
          num_params_per_child += np.prod([d.value for d in v.shape])
    print('Each child has {0} params'.format(num_params_per_child))

    self.batch_init_states = {
        'c': batch_prev_c,
        'h': batch_prev_h,
        'reset': batch_reset,
    }
    self.train_params = {
        'w_emb': w_emb,
        'w_0':w_0,
        'b': b,
        'w_lstm':  w_lstm,
        'w_soft': w_soft}

    self.eval_params = {
        'w_emb': w_emb,
        'w_0':w_0,
        'b': b,
        'w_lstm': w_lstm,
        'w_soft': w_soft,
    }

  def _forward(self, x, y, model_params, init_states, is_training=False):
    """Computes the logits.
    Args:
      x: [batch_size, num_steps], input batch.
      y: [batch_size, num_steps], output batch.
      model_params: a `dict` of params to use.
      init_states: a `dict` of params to use.
      is_training: if `True`, will apply regularizations.
    Returns:
      loss: scalar, cross-entropy loss
    """
    print("entered the forward function")
    w_emb = model_params['w_emb']
    w_lstm = model_params['w_lstm']
    w_soft = model_params['w_soft']
    w_0 = model_params['w_0']
    prev_h = init_states['h']
    prev_c = init_states['c']
    b = model_params['b']

    print("X before embedding {}".format(x))
    print("emb before embedding {}".format(w_emb))
    emb = tf.nn.embedding_lookup(w_emb, x)
    print("emb_befoer_LSTM = {}".format(emb))
    batch_size = self.params.batch_size
    hidden_size = self.params.hidden_size
    sample_arc = self.sample_arc

    print("emb_size :{}".format(emb.get_shape()))

    print("before using LSTM")
    out_c, out_h, all_h, var_h = _lstm(sample_arc, emb, prev_c,
                                       prev_h, w_0, w_lstm, params= self.params)
    prev_c , prev_h = out_c, out_h
    print("prev_c , pre_h, out_h , out_c = {}, {}, {}, {}".format(prev_c, prev_h, out_h, out_c))
    carry_on = [prev_h + prev_c]
    print("after using LSTM")
    top_h = all_h
    print("top_h size = {}".format(top_h))

    #logits = tf.einsum('bnh,vh->bnv', top_h, w_soft) #Linear layer after the LSTM like h * w_softmax
    #out = [b ,h]
    #w_soft = [h , v]
    logits = tf.matmul(out_h , w_soft) + b
    print("logits = {}".format(logits))
    print("y = {}".format(y))
    #y = tf.reshape(y, [batch_size])
    print("y = {}".format(y))
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y[:,-1],
                                                          logits=logits)
    loss = tf.reduce_mean(loss)
    reg_loss = loss  # `loss + regularization_terms` is for training only
    if is_training:
      #L2 weight reg
      #self.l2_reg_loss = tf.add_n([tf.nn.l2_loss(w**2) for w in tf.trainable_variables()])
      reg_loss += self.params.weight_decay*tf.add_n([tf.reduce_sum(w ** 2) for w in tf.trainable_variables()])

      #activation L2 reg
      #reg_loss += self.params.alpha * tf.add_n([tf.reduce_sum(h ** 2) for h in all_h])

      #activation slowness L2 reg
      #reg_loss += self.params.beta * tf.add_n([tf.reduce_mean((h[:, 1:, :] - h[:, :-1, :]) ** 2) for h in all_h])

    with tf.control_dependencies(carry_on):
      self.loss = tf.identity(loss)
      if is_training:
        self.reg_loss = tf.identity(reg_loss)

    #self.loss = tf.identity(loss)
    #self.l2_reg_loss = tf.identity(loss)
    self.l2_reg_loss = loss

    return reg_loss, loss

  def _build_train(self):
    """Build training ops."""
    print('-' * 80)
    print('Building train graph')
    print("line 366")
    reg_loss, loss = self._forward(self.x_train, self.y_train,
                                   self.train_params, self.batch_init_states,
                                   is_training=True)
    print("line 370")
    tf_vars = [v for v in tf.trainable_variables()
               if v.name.startswith(self.name)]
    global_step = tf.train.get_or_create_global_step()
    lr_scale = (tf.cast(tf.shape(self.y_train)[-1], dtype=tf.float32) /
                tf.cast(self.params.bptt_steps, dtype=tf.float32))
    learning_rate = utils.get_lr(global_step, self.params) * lr_scale
    if self.params.grad_bound:
      grads = tf.gradients(reg_loss, tf_vars)
      clipped_grads, grad_norm = tf.clip_by_global_norm(grads,
                                                        self.params.grad_bound)
    print("line 381")

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    train_op = optimizer.apply_gradients(zip(clipped_grads, tf_vars),
                                         global_step=global_step)

    self.train_loss = loss
    self.train_op = train_op
    self.grad_norm = grad_norm
    self.learning_rate = learning_rate

  def _build_valid(self):
    print('Building valid graph')
    _, loss = self._forward(self.x_valid, self.y_valid,
                            self.eval_params, self.batch_init_states)
    self.valid_loss = loss
    self.rl_loss = loss

  def eval_valid(self, sess):
    """Eval 1 round on valid set."""
    total_loss = 0
    for _ in range(self.num_valid_batches):
      sess.run(self.batch_init_states['reset'])
      total_loss += sess.run(self.valid_loss)
    valid_ppl = np.exp(total_loss / self.num_valid_batches)
    print('valid_ppl={0:<.2f}'.format(valid_ppl))

    return valid_ppl