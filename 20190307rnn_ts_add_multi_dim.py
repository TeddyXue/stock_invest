import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import data_loader
import matplotlib.pyplot as plt

class SeriesPredictor:

    def __init__(self, input_dim, seq_size, hidden_dim):
        # Hyperparameters
        self.input_dim = input_dim
        self.seq_size = seq_size
        self.hidden_dim = hidden_dim

        # Weight variables and input placeholders
        self.W_out = tf.Variable(tf.random_normal([hidden_dim, input_dim]), name='W_out')
        self.b_out = tf.Variable(tf.random_normal([input_dim]), name='b_out')
        self.x = tf.placeholder(tf.float32, [None, seq_size, input_dim])
        self.y = tf.placeholder(tf.float32, [None, input_dim])

        # Cost optimizer
        self.cost = tf.reduce_mean(tf.square(self.model() - self.y))
        # self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y,logits=self.model_traval()))
        self.train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.cost)

        # Auxiliary ops
        self.saver = tf.train.Saver()
        self.path='./model_traval/'
    def model(self):
        """
        :param x: inputs of size [T, batch_size, input_size]
        :param W: matrix of fully-connected output layer weights
        :param b: vector of fully-connected output layer biases
        """
        cell = rnn.BasicLSTMCell(self.hidden_dim)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell,output_keep_prob=0.8)
        outputs, final_state = tf.nn.dynamic_rnn(cell, self.x, dtype=tf.float32)
        results = tf.matmul(final_state[1], self.W_out) + self.b_out
        return results

    def train(self, train_x, train_y, test_x, test_y):
        with tf.Session() as sess:
            tf.get_variable_scope().reuse_variables()
            sess.run(tf.global_variables_initializer())
            step = 0
            for i in range(200):
                train_err = sess.run([self.train_op, self.cost], feed_dict={self.x: train_x, self.y: train_y})
                if step % 100 == 0:
                    test_err = sess.run(self.cost, feed_dict={self.x: test_x, self.y: test_y})
                    print('step: {}\t\ttrain err: {}\t\ttest err: {}\n'.format(step, train_err[-1], test_err))

                step += 1
            save_path = self.saver.save(sess, self.path)
            print('Model saved to {}'.format(save_path))

    def re_train(self, train_x, train_y, test_x, test_y):
        with tf.Session() as sess:
            tf.get_variable_scope().reuse_variables()
            self.saver.restore(sess, self.path)
            # sess.run(tf.global_variables_initializer())
            step = 0
            for i in range(1001):
                train_err = sess.run([self.train_op, self.cost], feed_dict={self.x: train_x, self.y: train_y})
                if step % 100 == 0:
                    test_err = sess.run(self.cost, feed_dict={self.x: test_x, self.y: test_y})
                    print('step: {}\t\ttrain err: {}\t\ttest err: {}\n'.format(step, train_err[-1], test_err))

                step += 1
            save_path = self.saver.save(sess, self.path)
            print('Model saved to {}'.format(save_path))


    def test(self, sess, test_x):
        tf.get_variable_scope().reuse_variables()
        self.saver.restore(sess, self.path)
        output = sess.run(self.model(), feed_dict={self.x: test_x})
        return output

def plot_results(train_x, predictions, actual, filename):
    plt.figure()
    num_train = len(train_x)
    plt.plot(list(range(num_train)), train_x, color='b', label='training data')
    plt.plot(list(range(num_train + len(actual), num_train + len(actual) + len(predictions))), predictions, color='r', label='predicted')
    plt.plot(list(range(num_train, num_train + len(actual))), actual, color='g', label='test data')
    plt.legend()
    plt.show()
if __name__ == '__main__':
    mode=2 #0训练 1继续训练 2看结果

    seq_size = 60
    predictor = SeriesPredictor(input_dim=5, seq_size=seq_size, hidden_dim=50)
    data = data_loader.load_series('data_test.txt')
    train_data, actual_vals, sample= data_loader.split_data(data,seq_size)
    # print(train_data)
    # print(np.shape(train_data))
    train_x, train_y = [], []
    for i in range(len(train_data) - seq_size - 1):
        train_x.append(train_data[i:i+seq_size])
        train_y.append(train_data[i+seq_size+1])
    # print(np.shape(train_x),np.shape(train_y))
    # print(train_y)
    # print(np.shape(train_y))
    test_x, test_y = [], []
    for i in range(len(actual_vals) - seq_size - 1):
        test_x.append(actual_vals[i:i+seq_size])
        test_y.append(actual_vals[i+seq_size+1])

        #开始训练
    if mode == 0:
        predictor.train(train_x, train_y, test_x, test_y)

        #看结果


    else:
        if mode == 1:
            predictor.re_train(train_x, train_y, test_x, test_y)
        else:
            with tf.Session() as sess:
                sam=[sample]
                # print(sam[-1][-1][4])
                predicted_vals = predictor.test(sess, sam)[-1]
                print(predicted_vals)
                # high=predicted_vals[0] - sam[-1][-1][4]
                # low=predicted_vals[1] - sam[-1][-1][4]
                open=predicted_vals[2] - sam[-1][-1][4]
                value=predicted_vals[3] - sam[-1][-1][3]
                close=predicted_vals[4] - sam[-1][-1][4]
                # print('HIGH:','%.2f%%'%(high*100),"\t",'LOW:','%.2f%%'%(low*100),"\t",'OPEN:','%.2f%%'%(open*100),"\t",'VALUE:','%.2f%%'%(value*100),"\t",'CLOSE:','%.2f%%'%(close*100))
                print('OPEN:','%.2f%%'%(open*100),"\t",'VALUE:','%.2f%%'%(value*100),"\t",'CLOSE:','%.2f%%'%(close*100))
