import tensorflow as tf
import numpy as np
import data_utils
import utils
import time
import os
import pickle
import sys
import time
tf.enable_eager_execution()
tf.executing_eagerly()

flags = tf.app.flags
gfile = tf.gfile
FLAGS = flags.FLAGS
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
flags.DEFINE_boolean('reset_output_dir', False, '')
flags.DEFINE_string('output_dir', None, '')
flags.DEFINE_string('data_path', None, '')
flags.DEFINE_integer('log_every', 200, '')
data_path = r'F:\workspace\enas_lm\src\ptb.pkl'
output_dir = r'F:\workspace\enas_lm\src\output_dir'

### Adding hparams
batch_size = 16
hidden_size = 128
num_of_steps = 25
input_size = 4
best_valid_ppl_threshold = 5
bptt_steps = 35
grad_bound = 0.1
init_range = 0.04
learning_rate = 1.
num_train_epochs = 600
vocab_size = 10000
weight_decay = 8e-7
start_decay_epoch = 10
decay_every_epoch = 1
num_train_steps = 0
emb_size = 400
num_train_batches = 0


def lstm(x, h_prev, ct_prev, w_h, w_x, b):
    """
    x: input sequence [batch_size x input_size]
    h_prev: previous hidden state
    ct_prev: previous cell state (has to be initialized before the loop)
    w_h: weights of hidden_states [hidden_size x hidden_size]
    w_x: input weights [input_size x 4 * hidden_size]
    b : biases [batch_size x 4 * hidden_size]
    :return:
    next_h: next hidden state
    next_c: next cell component
    """
    num_steps = tf.shape(x)[1]
    all_h = tf.TensorArray(dtype=tf.float32, size=num_steps, infer_shape=False)
    next_h, next_c = [], []

    def _condition(step, *unused_args):
        return tf.less(step, num_steps)

    x = x[:, 1, :]
    wxi, wxf, wxg, wxo = tf.split(w_x, 4, 1)
    whi, whf, whg, who = tf.split(w_h, 4, 1)
    bi, bf, bg, bo = tf.split(b, 4, 1)

    def _body(step, ht_prev, all_ht, ct_prev):

        i = tf.sigmoid(tf.matmul(x, wxi) + tf.matmul(ht_prev, whi) + bi)
        f = tf.sigmoid(tf.matmul(x, wxf) + tf.matmul(ht_prev, whf) + bf)
        g = tf.tanh(tf.matmul(x, wxg) + tf.matmul(ht_prev, whg) + bg)
        o = tf.sigmoid(tf.matmul(x, wxo) + tf.matmul(ht_prev, who) + bo)

        ct = i * g + f * ct_prev
        ht = tf.multiply(o, tf.tanh(ct))
        all_h = all_ht.write(step, ht)
        next_c.append(ct)
        next_h.append(ht)
        return step+1, ht, all_h, ct

    loop_inps = [tf.constant(0, dtype=tf.int32), h_prev, all_h, ct_prev]
    _, ht, all_h, ct= tf.while_loop(cond= _condition,
                                    body= _body,
                                    loop_vars= loop_inps)
    all_h = tf.transpose(all_h.stack(), [1, 0, 2])

    return ht, all_h, ct


class LM(object):
    print('-' * 80)
    print('Building LM')

    def __init__(self, x_train, x_valid):
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_of_steps = num_of_steps
        self.input_size= input_size

        (self.x_train, self.y_train,
         self.num_train_batches, self.reset_start_idx,
         self.should_reset, self.base_bptt) = data_utils.input_producer(
            x_train, batch_size, bptt_steps, random_len=True)

        num_train_steps = self.num_train_batches * num_train_epochs

        (self.x_valid, self.y_valid,
         self.num_valid_batches) = data_utils.input_producer(
            x_valid, batch_size, bptt_steps)

        self.build_params()
        self.build_train()
        self.build_valid()

    def build_params(self):
        print('-' * 80)
        print('Building model params')
        initializer = tf.initializers.random_uniform(minval=-init_range,
                                                     maxval=init_range)

        '''
        Weights: 
        w_x = input weights matrix: 4 * input_size, hidden_size
        w_h = hidden state weights matrix: 4 * hidden_size, hidden_size
        '''
        emb_w = tf.get_variable('emb', [vocab_size, input_size], initializer = initializer)
        w_x = tf.get_variable("w_x", [input_size, 4 * hidden_size], initializer= initializer)
        w_h = tf.get_variable("w_h", [hidden_size, 4 * hidden_size], initializer= initializer)
        b = tf.get_variable("b", [batch_size, 4 * hidden_size], initializer= initializer)
        #w_ct = tf.get_variable("w_ct", [hidden_size, hidden_size], initializer = initializer)

        init_shape =[batch_size , hidden_size]
        batch_prev_c = tf.get_variable('c', init_shape, dtype=tf.float32, trainable=False)
        batch_prev_h = tf.get_variable('h', init_shape, dtype=tf.float32, trainable=False)

        zeros = np.zeros(init_shape, dtype=np.float32)
        batch_reset = tf.assign(batch_prev_h[-1], zeros)
        batch_reset = tf.assign(batch_prev_c[-1], zeros)

        softmax_w = tf.get_variable('softmax_w',shape=  [hidden_size, vocab_size], dtype=tf.float32, initializer = initializer)
        softmax_b = tf.get_variable('softmax_b', shape= [vocab_size], dtype=tf.float32,  initializer = initializer)


        self.batch_init_states = {
            'c': batch_prev_c,
            'h': batch_prev_h,
            'reset': batch_reset,
        }

        self.train_params= {
            'w_emb': emb_w,
            'w_x': w_x,
            'w_h': w_h,
            'b': b,
            'prev_ct': batch_prev_c,
            'softmax_w': softmax_w,
            'softmax_b': softmax_b,
        }

        self.eval_params= {
            'w_emb': emb_w,
            'w_x': w_x,
            'w_h': w_h,
            'b': b,
            'prev_ct': batch_prev_c,
            'softmax_w': softmax_w,
            'softmax_b': softmax_b,
        }


        num_params = sum([np.prod(v.shape) for v in tf.trainable_variables()])
        print("LSTM has {} num of params".format(num_params))

    def _forward(self, x, y, model_params, init_states, is_training= False):
        """
        :param x: [batch_size , num_steps] input batch
        :param y: [batch_size , num_steps] output batch
        :param model_params: a dict of params to use
        :param init_states: a dict of params to use
        :param is_training: if True, will apply regulaization
        :return: loss: scalar, cross_entropy loss
        """
        w_emb = model_params['w_emb']
        w_x = model_params['w_x']
        w_h = model_params['w_h']
        softmax_w = model_params['softmax_w']
        b = model_params['b']
        #softmax_w = model_params['softmax_w']
        softmax_b = model_params['softmax_b']

        prev_h = init_states['h']
        prev_ct = init_states['c']

        emb = tf.nn.embedding_lookup(w_emb, x)
        ht, all_h, ct = lstm(emb, prev_h, prev_ct, w_h, w_x, b)
        prev_h, prev_ct = ht, ct

        print('size of ht before reshape= {}'.format(ht.get_shape()))
        # linear layer which size is h_s * vocab_size
        ht = tf.reshape(ht, [-1, hidden_size])
        ht = tf.nn.xw_plus_b(ht, softmax_w, softmax_b)
        #ht = tf.reshape(ht, [batch_size, num_of_steps, vocab_size])
        print('size of ht after linear layer= {}'.format(ht.get_shape()))
        print('size of y= {}'.format(y.get_shape()))

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels= y, logits= all_h)
        loss = tf.reduce_mean(loss)

        return loss

    def build_train(self):
        print('Building training graph')
        loss = self._forward(self.x_train, self.y_train, self.train_params, self.batch_init_states, is_training=True)
        tf_vars = [v for v in tf.trainable_variables()]
        global_step = tf.train.get_or_create_global_step()
        lr = learning_rate * 0.98
        if grad_bound:
            grads = tf.gradients(loss, tf_vars)
            clipped_grads, grad_norm = tf.clip_by_global_norm(grads, grad_bound)

        optimizer = tf.train.GradientDescentOptimizer(lr)
        train_op = optimizer.apply_gradients(zip(clipped_grads, tf_vars), global_step=global_step)

        self.train_loss = loss
        self.train_op = train_op
        self.grad_norm = grad_norm
        self.learning_rate = lr


    def build_valid(self):
        print("Building validation graph")
        _, loss = self._forward(self.x_valid, self.y_valid, self.eval_params, self.batch_init_states)
        self.valid_loss = loss

    def eval_valid(self, sess):
        print("Evaluation")
        total_loss = 0
        for _ in range(self.num_valid_batches):
            sess.run(self.batch_init_states['reset'])
            total_loss += sess.run(self.valid_loss)
        valid_ppl = np.exp(total_loss/self.num_valid_batches)
        print('valid_ppl = {0:<.2f}'.format(valid_ppl))
        return valid_ppl

def get_ops(x_train, x_valid):
    lm = LM(x_train, x_valid)
    num_train_batches = lm.num_train_batches
    ops = {
        'train_op': lm.train_op,
        'learning_rate': lm.learning_rate,
        'grad_norm': lm.grad_norm,
        'trian_loss': lm.train_loss,
        'global_step': tf.train.get_or_create_global_step(),
        'reset_batch_states': lm.batch_init_states['reset'],
        'eval_valid': lm.eval_valid,
        'reset_start_idx': lm.reset_start_idx,
        'should_reset': lm.should_reset,
    }
    return ops


def train():
    with gfile.GFile(data_path, 'rb') as finp:
        x_train, x_valid, x_test, _, _= pickle.load(finp)
        print("x_valid = {}".format(x_valid))

    g = tf.Graph()
    with g.as_default():
        ops = get_ops(x_train, x_valid)
        run_ops=[
            ops['train_loss'],
            ops['grad_norm'],
            ops['learning_rate'],
            ops['should_reset'],
            ops['train_op']
        ]

        sess = tf.Session(graph= g)
        accum_loss= 0
        accum_step = 0
        epoch = 0
        best_valid_ppl = []
        start_time = time.time()

        while True:
            sess.run(ops['reset_batch_states'])
            loss, grad_norm, learning_rate, should_reset, train_op = sess.run(run_ops)
            accum_loss += loss
            accum_step += 1
            step = sess.run(ops['global_step'])
            if step % 10 == 0:
                train_ppl = np.exp(accum_loss/accum_step)
                log_string = 'epoch={0:<5d}'.format(epoch)
                log_string = 'step={0:<7d}'.format(step)
                log_string = 'ppl={0:<9.2f}'.format(train_ppl)
                log_string = 'loss={0:<9.2f}'.format(loss)

            if should_reset:
                epoch +=1
                accum_step= 0
                accum_loss= 0
                valid_ppl = ops['eval_valid'](sess)
                sess.run([ops['reset_batch_states'], ops['reset_start_idx']])

            sess.close()

def main(unused_args):
    output_dir = r'F:\workspace\enas_lm\src\output_dir'
    log_file = os.path.join(output_dir, 'stdout')
    sys.stdout = utils.Logger(log_file)
    train()

if __name__ == '__main__':
    tf.app.run()