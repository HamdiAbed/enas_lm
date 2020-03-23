import tensorflow as tf
import numpy as np
import data_utils
import utils
import time
import os
import pickle
import sys

flags = tf.app.flags
gfile = tf.gfile
FLAGS = flags.FLAGS

flags.DEFINE_boolean('reset_output_dir', False, '')
flags.DEFINE_string('output_dir', None, '')
flags.DEFINE_string('data_path', None, '')

flags.DEFINE_integer('log_every', 200, '')
#TODO: define the rnn model
from tensorflow.python.training.training_util import global_step

global emb

def lstm(x, prev_h, w_prev, w_skip, params):

    batch_size= x.get_shape()[0].value
    num_of_steps = tf.shape(x)[1]
    num_layers = 6
    emb = x
    all_h = tf.TensorArray(dtype=tf.float32, size= num_of_steps, infer_shape=False)

    u_skip = []
    #Creating the weights matrix or all layers
    for layer_id in range(num_layers):
        u_skip.append(w_skip[layer_id])
    w_skip = u_skip
    vars = [w_prev] + w_skip  #Trainiable parameters

    def _condition(step, *unused_args):
        return tf.less(step,num_of_steps)

    def _body(step , prev_h, all_h):
        inp = emb[:, step, :]

        #define the h and x
        #h = prev_h * w_h
        #x = input * w_x
        h = tf.matmul(prev_h, tf.split(w_prev, 2, axis=1)[0])
        x = tf.matmul(inp, tf.split(w_prev, 2, axis=1)[1])

        #perform RNN operation with s = prev_h + x*(h-prev_h)
        h = tf.tanh(h)
        x = tf.sigmoid(x)
        s = prev_h + x *(h-prev_h)
        s.set_shape([batch_size, params.hidden_size])
        layers_inp = [s]
        prev_c = s   #cell_state
        layers = []

        for layer_id in range(num_layers):
            #first 4 nodes are input to the later layers
            if layer_id < 4:
                h = layers_inp[0]
                layers.append(h)

            elif layer_id == 4:
                layers.append(prev_c)

            else:
                # here should be the sampling of connections and operations but I will just set it to id operation
                h.set_shape([batch_size, params.hidden_size])
                layers.append(h)

        next_h = tf.add_n(layers[1:])/tf.cast(num_layers, dtype=tf.float32) #next_h is an average of all deadends (as per ENAS approach)
        all_h = all_h.write(step, next_h)
        return step + 1, next_h, all_h

    loop_inps = [tf.constant(0, dtype= tf.int32), prev_h, all_h]
    _, next_h, all_h = tf.while_loop(_condition, _body, loop_inps)
    all_h = tf.transpose(all_h.stack(), [1,0,2])
    return  next_h, all_h, vars

def _set_default_params(params):
    """Set default hyper-parameters."""
    params.add_hparam('alpha', 0.0)  # activation L2 reg
    params.add_hparam('beta', 1.)  # activation slowness reg
    params.add_hparam('best_valid_ppl_threshold', 5)
    params.add_hparam('grad_bound', 0.1)
    params.add_hparam('hidden_size', 200)
    params.add_hparam('init_range', 0.04)
    params.add_hparam('learning_rate', 1.)
    params.add_hparam('num_train_epochs', 600)
    params.add_hparam('vocab_size', 10000)
    params.add_hparam('weight_decay', 8e-7)
    params.add_hparam('batch_size', 128)
    params.add_hparam('bptt_steps', 35)
    return params

#TODO: def the LM class
class LM(object):
    """Language Model"""

    def __init__(self, params, x_train, x_valid, name):
        print('-' * 80)
        print('Building LM')

        self.params= _set_default_params(params)
        self.name = name

        #Training data
        (self.x_train, self.y_train,
         self.num_train_batches, self.reset_start_idx,
         self.should_reset, self.base_bptt) = data_utils.input_producer(
            x_train, params.batch_size, params.bptt_steps, random_len=True)

        params.add_hparam('num_train_steps', self.num_train_batches * params.num_train_epochs)

        # valid data
        (self.x_valid, self.y_valid,
         self.num_valid_batches) = data_utils.input_producer(
            x_valid, params.batch_size, params.bptt_steps)

        self._build_params()
        self._build_train()
        self._build_valid()

    def _build_params(self):
        """"Create model params"""
        print('-' * 80)
        print('Building model params')

        initializer = tf.initializers.random_uniform(minval=-self.params.init_range,
                                                     maxval=self.params.init_range)
        num_of_layers = 6
        hidden_size = self.params.hidden_size
        batch_size = self.params.batch_size
        with tf.variable_scope(self.name, initializer=initializer):
            with tf.variable_scope('embedding'):
                w_emb = tf.get_variable('w', [self.params.vocab_size, hidden_size], initializer= initializer)

            with tf.variable_scope('lstm'):
                w_prev = tf.get_variable('w_prev', [hidden_size, 2 * hidden_size], initializer= initializer)

                w_skip = []
                for layer_id in range(num_of_layers):
                    with tf.variable_scope('layer_{}'.format(layer_id)):
                        w = tf.get_variable(
                            'w', [layer_id, hidden_size, hidden_size], initializer=initializer)
                        w_skip.append(w)

            with tf.variable_scope('init_states'):
                with tf.variable_scope('batch'):
                    init_shape = [self.params.batch_size, hidden_size]
                    batch_prev_h = tf.get_variable(
                        's', init_shape, dtype=tf.float32, trainable=False)
                    zeros = np.zeros(init_shape, dtype=np.float32)
                    batch_reset = tf.assign(batch_prev_h, zeros)

        self.num_params = sum([np.prod(v.shape) for v in tf.trainable_variables()
                               if v.name.startswith(self.name)]).value
        print('All children have {0} params'.format(self.num_params))

        self.batch_init_states = {
            'h': batch_prev_h,
            'reset': batch_reset,
        }
        self.train_params = {
            'w_emb': w_emb,
            'w_prev': w_prev,
            'w_skip': w_skip,
            'w_soft': w_emb}

        self.eval_params = {
            'w_emb': w_emb,
            'w_prev': w_prev,
            'w_skip': w_skip,
            'w_soft': w_emb,
        }

    def _forward(self, x, y, model_params, init_states, is_training = False):
        "To comute the logits"

        w_emb = model_params['w_emb']
        w_prev = model_params['w_prev']
        w_skip = model_params['w_skip']
        w_soft = model_params['w_soft']
        prev_h = init_states['h']

        emb = tf.nn.embedding_lookup(w_emb, x)
        batch_size = self.params.batch_size
        hidden_size = self.params.hidden_size

        out_s, all_h, var_s = lstm(emb, prev_h, w_prev, w_skip,
                                      params=self.params)
        top_s = all_h
        carry_on = [tf.assign(prev_h, out_s)]
        logits = tf.einsum('bnh,vh->bnv', top_s, w_soft)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                              logits=logits)
        loss = tf.reduce_mean(loss)

        reg_loss = loss  # `loss + regularization_terms` is for training only
        if is_training:
            # L2 weight reg
            self.l2_reg_loss = tf.add_n([tf.nn.l2_loss(w ** 2) for w in var_s])
            reg_loss += self.params.weight_decay * self.l2_reg_loss

            # activation L2 reg
            reg_loss += self.params.alpha * tf.reduce_mean(all_h ** 2)

            # activation slowness reg
            reg_loss += self.params.beta * tf.reduce_mean(
                (all_h[:, 1:, :] - all_h[:, :-1, :]) ** 2)

        # with tf.control_dependencies(carry_on):
        loss = tf.identity(loss)
        if is_training:
            reg_loss = tf.identity(reg_loss)

        return reg_loss, loss

    def _build_train(self):
        """Build training ops."""
        print('-' * 80)
        print('Building train graph')
        reg_loss, loss = self._forward(self.x_train, self.y_train,
                                       self.train_params, self.batch_init_states,
                                       is_training=True)

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

global_step= tf.train.get_or_create_global_step()

def get_ops(params, x_train, x_valid):
  """Build [train, valid, test] graphs."""
  lm = LM(params, x_train, x_valid, 'child')
  params.add_hparam('num_train_batches', lm.num_train_batches)
  ops = {
      'train_op': lm.train_op,
      'learning_rate': lm.learning_rate,
      'grad_norm': lm.grad_norm,
      'train_loss': lm.train_loss,
      'l2_reg_loss': lm.l2_reg_loss,
      'global_step': tf.train.get_or_create_global_step(),
      'reset_batch_states': lm.batch_init_states['reset'],
      'eval_valid': lm.eval_valid,

      'reset_start_idx': lm.reset_start_idx,
      'should_reset': lm.should_reset,
  }
  return ops

def net_train(params):
    tf.reset_default_graph()

    with gfile.GFile(params.data_path, 'rb') as finp:
        x_train, x_valid, _, _, _ = pickle.load(finp)
        print('-' * 80)
        print('train_size: {0}'.format(np.size(x_train)))
        print('valid_size: {0}'.format(np.size(x_valid)))

    g = tf.Graph()
    with g.as_default():
        ops = get_ops(params, x_train, x_valid)
        run_ops = [
            ops['train_loss'],
            ops['l2_reg_loss'],
            ops['grad_norm'],
            ops['learning_rate'],
            ops['should_reset'],
            ops['train_op'],
        ]

        sess = tf.Session()
        accum_loss= 0
        accum_step = 0
        epoch = 0
        best_valid_ppl = []
        start_time = time.time()

        while True:
            try:
                loss, l2_reg, gn, lr, should_reset, _ = sess.run(run_ops)
                # print('loss type is {}'.format(type(loss)))
                accum_loss += loss
                accum_step += 1
                step = sess.run(global_step)

                if step % params.log_every == 0:
                    train_ppl = np.exp(accum_loss / accum_step)
                    mins_so_far = (time.time() - start_time) / 60.
                    log_string = 'epoch={0:<5d}'.format(epoch)
                    log_string += ' step={0:<7d}'.format(step)
                    log_string += ' ppl={0:<9.2f}'.format(train_ppl)
                    log_string += ' lr={0:<7.2f}'.format(lr)
                    log_string += ' |w|={0:<6.2f}'.format(l2_reg)
                    log_string += ' |g|={0:<6.2f}'.format(gn)
                    log_string += ' mins={0:<.2f}'.format(mins_so_far)
                    print(log_string)

                if should_reset:
                    epoch += 1
                    accum_loss = 0
                    accum_step = 0
                    valid_ppl = ops['eval_valid'](sess)
                    sess.run([ops['reset_batch_states'], ops['reset_start_idx']])
                    best_valid_ppl.append(valid_ppl)

                if step >= params.num_train_steps:
                    break

            except tf.errors.InvalidArgumentError:
                break

        sess.close()
def main(unused_args):
    data_path = r'H:\ubuntu_fiels\enas-rnn-search\src\ptb.pkl'
    params = tf.contrib.training.HParams(
        data_path=data_path,
        log_every=20,
    )

    net_train(params)

if __name__ == '__main__':
    tf.app.run()