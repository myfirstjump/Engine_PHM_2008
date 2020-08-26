import pandas as pd

class DataTraining(object):

    def __init__(self):
        pass

    def training_PHM_2008_Engine_data(self, model_string):

        if model_string == 'RNN':

class RnnInTF18(object):
    def __init__(self):
        BATCH_START = 0
        TIME_STEPS = 20
        BATCH_SIZE = 50
        INPUT_SIZE = 1
        OUTPUT_SIZE = 1
        CELL_SIZE = 32
        LR = 0.006
    
    def get_batch():
        global BATCH_START, TIME_STEPS
        # xs shape (50batch, 20steps)
        xs = np.arange(BATCH_START, BATCH_START+TIME_STEPS*BATCH_SIZE).reshape((BATCH_SIZE, TIME_STEPS)) / (10*np.pi)
        seq = np.sin(xs)

        # privide same pattern as sin
        res = 0.3 * np.square(np.cos(xs)) - 0.5 * np.square(np.sin(xs)) + 0.3 * np.cos(xs) - 0.1 * np.sin(xs)

        # given sin is working too
        # res = np.sin(xs)

        BATCH_START += TIME_STEPS
        # plt.plot(xs[0, :], res[0, :], 'r', xs[0, :], seq[0, :], 'b--')
        # plt.show()
        # returned seq, res and xs: shape (batch, step, input)
        return [seq[:, :, np.newaxis], res[:, :, np.newaxis], xs]

    class LSTMRNN(object):
        def __init__(self, n_steps, input_size, output_size, cell_size, batch_size):
            self.n_steps = n_steps
            self.input_size = input_size
            self.output_size = output_size
            self.cell_size = cell_size
            self.batch_size = batch_size
            with tf.name_scope('inputs'):
                self.xs = tf.placeholder(tf.float32, [None, n_steps, input_size], name='xs')
                self.ys = tf.placeholder(tf.float32, [None, n_steps, output_size], name='ys')
            with tf.variable_scope('in_hidden'):
                self.add_input_layer()
            with tf.variable_scope('LSTM_cell'):
                self.add_cell()
            with tf.variable_scope('out_hidden'):
                self.add_output_layer()
            with tf.name_scope('cost'):
                self.compute_cost()
            with tf.name_scope('train'):
                self.train_op = tf.train.AdamOptimizer(LR).minimize(self.cost)

        def add_input_layer(self,):
            l_in_x = tf.reshape(self.xs, [-1, self.input_size], name='2_2D')  # (batch*n_step, in_size)
            # Ws (in_size, cell_size)
            Ws_in = self._weight_variable([self.input_size, self.cell_size])
            # bs (cell_size, )
            bs_in = self._bias_variable([self.cell_size,])
            # l_in_y = (batch * n_steps, cell_size)
            with tf.name_scope('Wx_plus_b'):
                l_in_y = tf.matmul(l_in_x, Ws_in) + bs_in
            # reshape l_in_y ==> (batch, n_steps, cell_size)
            self.l_in_y = tf.reshape(l_in_y, [-1, self.n_steps, self.cell_size], name='2_3D')

        def add_cell(self):
            lstm_cell = rnn.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True)
            with tf.name_scope('initial_state'):
                self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
            self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
                lstm_cell, self.l_in_y, initial_state=self.cell_init_state, time_major=False)

        def add_output_layer(self):
            # shape = (batch * steps, cell_size)
            l_out_x = tf.reshape(self.cell_outputs, [-1, self.cell_size], name='2_2D')
            Ws_out = self._weight_variable([self.cell_size, self.output_size])
            bs_out = self._bias_variable([self.output_size, ])
            # shape = (batch * steps, output_size)
            with tf.name_scope('Wx_plus_b'):
                self.pred = tf.matmul(l_out_x, Ws_out) + bs_out

        def compute_cost(self):
            with tf.name_scope('average_cost'):
                self.cost = tf.div(
                    tf.reduce_sum(self.ms_error(
                    tf.reshape(self.pred, [-1], name='reshape_pred'),
                    tf.reshape(self.ys,   [-1], name='reshape_target')), name='losses_sum'),
                    self.batch_size,
                    name='average_cost')
                tf.summary.scalar('cost', self.cost)

        def ms_error(self, y_pre, y_target):
            return tf.square(tf.subtract(y_pre, y_target))

        def _weight_variable(self, shape, name='weights'):
            initializer = tf.random_normal_initializer(mean=0., stddev=1.,)
            return tf.get_variable(shape=shape, initializer=initializer, name=name)

        def _bias_variable(self, shape, name='biases'):
            initializer = tf.constant_initializer(0.1)
            return tf.get_variable(name=name, shape=shape, initializer=initializer)
            

if __name__ == '__main__':
    model = LSTMRNN(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE)
    sess = tf.Session()
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs", sess.graph)
    
    sess.run(tf.global_variables_initializer())
    # $ tensorboard --logdir='logs'

    plt.ion()
    plt.show()
    for i in range(200):
        seq, res, xs = get_batch()
        if i == 0:
            feed_dict = {
                    model.xs: seq,
                    model.ys: res,
                    # create initial state
            }
        else:
            feed_dict = {
                model.xs: seq,
                model.ys: res,
                model.cell_init_state: state    # use last state as the initial state for this run
            }

        _, cost, state, pred = sess.run(
            [model.train_op, model.cost, model.cell_final_state, model.pred],
            feed_dict=feed_dict)

        # plotting

	# print('xs:')
	# print(xs[0,:])
	# print('pred:')
	# print(pred.flatten()[:TIME_STEPS])

	# only the first 20 samples are ploted
        plt.plot(xs[0, :], res[0].flatten(), 'r', xs[0, :], pred.flatten()[:TIME_STEPS], 'b--')

	# entirely plot the all 50x20 samples
        # plt.plot(xs[:].flatten(), res[:].flatten(), 'r', xs[:].flatten(), pred[:].flatten(), 'b--')

        plt.ylim((-1.2, 1.2))
        plt.draw()
        plt.pause(0.3)

        if i % 20 == 0:
            print('cost: ', round(cost, 4))
            result = sess.run(merged, feed_dict)
            writer.add_summary(result, i)
import time

import numpy as np

import tensorflow as tf
from tensorflow import keras

from neural_design import NeuralCalculation, LossDesign
from plot_module import PlotDesign

class DataTraining(object):
    def __init__(self):
        self.neural_obj = NeuralCalculation()
        self.loss_obj = LossDesign()
        self.plot_obj = PlotDesign()

    def sys_show_execution_time(method):
        def time_record(*args, **kwargs):
            start_time = time.time()
            result = method(*args, **kwargs)
            end_time = time.time()
            execution_time = np.round(end_time - start_time, 3)
            print('Running function:', method.__name__, ' cost time:', execution_time, 'seconds.')
            return result
        return time_record
        
    def model_design(self, model_name, data, hyperparameters, graph_output_dir=None):

        if model_name == 'Autoencoder':
            
            origin_dim = hyperparameters['origin_dim']
            encoding_dim = hyperparameters['encoding_dim']


            input_img = Input(shape=(origin_dim,))
            encoded = Dense(12, activation='relu')(input_img)
            encoded = Dense(encoding_dim, activation='relu')(encoded)

            decoded = Dense(12, activation='relu')(encoded)
            decoded = Dense(origin_dim, activation='sigmoid')(decoded)
            

        if model_name == 'DNN':
            model = tf.keras.models.Sequential([
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation=tf.nn.relu),
                tf.keras.layers.Dense(256, activation=tf.nn.relu),
                tf.keras.layers.Dense(10, activation=tf.nn.softmax)
            ])
        
        if model_name == 'CNN':
            model = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(filters=64, input_shape=(28,28,1), kernel_size=(3,3), strides=1, padding='valid', activation=tf.nn.relu),
                tf.keras.layers.MaxPooling2D(2,2),
                tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=1, padding='valid', activation=tf.nn.relu),
                tf.keras.layers.MaxPooling2D(2,2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation=tf.nn.relu),
                tf.keras.layers.Dense(10, activation=tf.nn.softmax)
            ])

        if model_name == 'GAN':

            # Two kinds of modeling method:
            # 1. Separately define G_net and D_net    <-   We use this.
            # 2. Define the hidden layer involve G and D. Fix the G weigths when training D, and vice versa.
            
            seed = hyperparameters['seed']
            batch_size = hyperparameters['batch_size']
            X_dim = hyperparameters['X_dim']
            z_dim = hyperparameters['z_dim']
            h_dim = hyperparameters['h_dim']
            lam = hyperparameters['lam']
            n_disc = hyperparameters['n_disc']
            lr = hyperparameters['lr']

            tf.set_random_seed(seed)
            np.random.seed(seed)

            tf.reset_default_graph()

            X = tf.placeholder(tf.float32, shape=[None, X_dim])
            X_target = tf.placeholder(tf.float32, shape=[None, X_dim])
            z = tf.placeholder(tf.float32, shape=[None, z_dim])

            G_sample, G_var = self.neural_obj.generator(z) # 由 m = 32 vectors
            D_real_logits, D_var = self.neural_obj.discriminator(X, spectral_normed=False)
            D_fake_logits, _ = self.neural_obj.discriminator(G_sample, spectral_normed=False, reuse=True)

            D_loss, G_loss = self.loss_obj.gan_loss(D_real_logits, D_fake_logits, gan_type='GAN', relativistic=False)
            D_solver = (tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5)).minimize(D_loss, var_list=D_var)
            G_solver = (tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5)).minimize(G_loss, var_list=G_var)

            # z search
            z_optimizer = tf.train.AdamOptimizer(0.0001)
            z_r = tf.get_variable('z_update', [batch_size, z_dim], tf.float32)
            G_z_r, _ = self.neural_obj.generator(z_r, reuse=True)

            z_r_loss = tf.reduce_mean(tf.abs(tf.reshape(X_target, [-1, 28, 28, 1]) - G_z_r))
            z_r_optim = z_optimizer.minimize(z_r_loss, var_list=[z_r])

            sess = tf.Session()

            tensorboard_output = graph_output_dir + 'gan_graphs'
            writer = tf.summary.FileWriter(tensorboard_output, sess.graph)
            sess.run(tf.global_variables_initializer())

            model = {}
            model['sess'] = sess
            model['D_solver'] = D_solver
            model['D_loss'] = D_loss
            model['G_solver'] = G_solver
            model['G_loss'] = G_loss
            model['G_sample'] = G_sample
            model['X'] = X
            model['z'] = z

        return model



    @sys_show_execution_time
    def model_training(self, model, data):
        
        training_images, training_labels = data
        model.compile(optimizer = 'adam',#optimizer = tf.compat.v1.train.AdamOptimizer(),
                      loss =  'sparse_categorical_crossentropy',
                      metrics = ['accuracy'])
        callbacks = CallBack()
        model.fit(training_images, training_labels, epochs=15, callbacks=[callbacks])

        return model

    @sys_show_execution_time
    def gan_model_training(self, data, output_dir, model, hyperparameters=None):

        batch_size = hyperparameters['batch_size']
        X_dim = hyperparameters['X_dim']
        z_dim = hyperparameters['z_dim']
        h_dim = hyperparameters['h_dim']
        lam = hyperparameters['lam']
        n_disc = hyperparameters['n_disc']
        lr = hyperparameters['lr']

        sess = model['sess']
        D_solver = model['D_solver']
        D_loss = model['D_loss']
        G_solver = model['G_solver']
        G_loss = model['G_loss']
        G_sample = model['G_sample']
        X = model['X']
        z = model['z']

        start_time = time.time()
        for it in range(300000):
            for _ in range(n_disc):

                # First, fix generator G, and update discriminator D.
                X_mb, _ = data.train.next_batch(batch_size)

                _, D_loss_curr = sess.run(
                    [D_solver, D_loss],
                    feed_dict={X: X_mb, z: self.neural_obj.sample_z(batch_size, z_dim)}
                )
            
            # Second, fix discriminator, and update generator G.
            X_mb, _ = data.train.next_batch(batch_size)
            _, G_loss_curr = sess.run(
                [G_solver, G_loss],
                feed_dict={X: X_mb, z: self.neural_obj.sample_z(batch_size, z_dim)}
            )

            if it % 10000 == 0:
                print('Iter: {}; Cost Time: {:.4}; D loss: {:.4}; G_loss: {:.4}'.format(it, time.time() - start_time, D_loss_curr, G_loss_curr))
                
                samples = sess.run(G_sample, feed_dict={z: self.neural_obj.sample_z(16, z_dim)})
                fig = self.plot_obj.plot(samples)

                dest_path = output_dir + 'gan_output/'
                self.plot_obj.plot_saving(dest_path=dest_path, filename='gan_generator_{}_{}'.format('mnist', it), suffix='png')

class CallBack(tf.keras.callbacks.Callback):

    # Each epoch end, will call the method on_epoch_end
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc')>0.98):
            print('Reached enough accuracy so stop training...')
            self.model.stop_training = True
