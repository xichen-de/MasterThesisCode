"""
Script to train pretrain network for predicting status and reward.
"""
#  MIT License
#
#  Copyright 2021 Xi Chen
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import argparse
import datetime
import os
import time
from typing import List, Tuple

import numpy as np
import tensorflow as tf
import yaml
from src.tools.divide_files import ROOT_STR
from stable_baselines.common.tf_layers import mlp
from tensorflow.python.keras.utils import to_categorical

__author__ = "Xi Chen"
__copyright__ = "TUM Cyber-Physical System Group"
__credits__ = [""]
__email__ = "xi.chen@tum.de"
__status__ = "Development"


class PretrainNetwork:
    """
    Pretrain transition and reward network, using for actor model update in DDPGPlan.
    """

    def __init__(self, transition_network: List[int], reward_network: List[int]):
        """
        Initialize pretrain network. Set up optimizer and log.
        :param transition_network: list of how many neurons in each layer of transition network
        :param reward_network: list of how many neurons in each layer of reward network
        """
        self.accuracy = None
        self.global_step = None
        self.input_ph = None
        self.optimization_summaries = None
        self.optimizer = None
        self.reward = None
        self.reward_loss = None
        self.reward_ph = None
        self.status = None
        self.status_loss = None
        self.status_ph = None
        self.total_loss = None
        self.train_op = None

        self.setup_model(transition_network, reward_network)
        self.setup_optimizer()
        self.setup_log()

        self.variables = tf.trainable_variables()
        self.saver = tf.train.Saver(self.variables, save_relative_paths=True)

    def setup_model(self, transition_network: List[int], reward_network: List[int]):
        """
        Set up model.
        :param transition_network: list of how many neurons in each layer of transition network
        :param reward_network: list of how many neurons in each layer of reward network
        """
        # Input placeholder
        self.input_ph = tf.placeholder(tf.float32, shape=[None, 26], name='input_ph')
        self.status_ph = tf.placeholder(tf.float32, shape=[None, 3], name='status_ph')
        self.reward_ph = tf.placeholder(tf.float32, shape=[None, 1], name='reward_ph')

        # Transition network
        with tf.variable_scope(name_or_scope="transition", reuse=False):
            tr_h = mlp(self.input_ph, transition_network)
            self.status = tf.layers.dense(tr_h, 3, activation=None, name="pred_status")
        correct_pred = tf.equal(tf.argmax(self.status, 1), tf.argmax(self.status_ph, 1), name='correct_pred')
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

        # Reward network
        with tf.variable_scope(name_or_scope="reward", reuse=False):
            softmax_res = tf.nn.softmax(self.status)
            reward_input = tf.concat([self.input_ph, softmax_res], 1)
            rew_h = mlp(reward_input, reward_network)
            self.reward = tf.layers.dense(rew_h, 1, activation=None, name="pred_reward")

    def setup_optimizer(self):
        """
        Set up optimizer.
        """
        # Define loss
        self.status_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.status, labels=self.status_ph),
            name="status_loss")
        self.reward_loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.reward, predictions=self.reward_ph),
                                          name="reward_loss")
        self.total_loss = tf.add(self.status_loss, self.reward_loss, name="total_loss")

        self.optimizer = tf.train.AdamOptimizer(0.001)
        grad = tf.gradients(self.total_loss, tf.trainable_variables())
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.train_op = self.optimizer.apply_gradients(zip(grad, tf.trainable_variables()),
                                                       global_step=self.global_step)

    def setup_log(self):
        """
        Set up log.
        """
        status_loss_summary = tf.summary.scalar('status_loss_summary', self.status_loss)
        acc_summary = tf.summary.scalar('acc', self.accuracy)
        reward_loss_summary = tf.summary.scalar('reward_loss_summary', self.reward_loss)
        self.optimization_summaries = tf.summary.merge([
            status_loss_summary,
            acc_summary,
            reward_loss_summary
        ])

    def train(self, sess: tf.Session, network_input: np.ndarray, status_ground_truth: np.ndarray,
              reward_ground_truth: np.ndarray):
        """
        Train network for one batch and one step.
        :param sess: Tensorflow session
        :param network_input: input numpy array
        :param status_ground_truth: status numpy array
        :param reward_ground_truth: reward numpy array
        :return: status loss, reward loss, Tensorflow summary, global step
        """
        _, s_loss, r_loss, train_summary, current_global_step = sess.run(
            [self.train_op, self.status_loss, self.reward_loss, self.optimization_summaries, self.global_step],
            feed_dict={self.input_ph: network_input, self.status_ph: status_ground_truth,
                       self.reward_ph: reward_ground_truth})
        return s_loss, r_loss, train_summary, current_global_step

    def test(self, sess: tf.Session, network_input: np.ndarray, status_ground_truth: np.ndarray,
             reward_ground_truth: np.ndarray):
        """
        Test network for one batch and one step.
        :param sess: Tensorflow session
        :param network_input: input numpy array
        :param status_ground_truth: status numpy array
        :param reward_ground_truth: reward numpy array
        :return: status loss, reward loss, Tensorflow summary
        """
        s_loss, r_loss, test_summary = sess.run([self.status_loss, self.reward_loss, self.optimization_summaries],
                                                feed_dict={self.input_ph: network_input,
                                                           self.status_ph: status_ground_truth,
                                                           self.reward_ph: reward_ground_truth})
        return s_loss, r_loss, test_summary

    def predict(self, sess: tf.Session, network_input: np.ndarray):
        """
        Predict status logits and reward.
        :param sess: Tensorflow session
        :param network_input: input numpy array
        :return: status logits, reward
        """
        status, reward = sess.run([self.status, self.reward], feed_dict={self.input_ph: network_input})
        return status, reward

    def save_model(self, sess: tf.Session, model_name: str, save_path: str):
        """
        Save model.
        :param sess: Tensorflow session
        :param model_name: model name
        :param save_path: save path of the model
        """
        os.makedirs(os.path.join(save_path, model_name), exist_ok=True)
        self.saver.save(sess, os.path.join(save_path, model_name, 'model', 'pretrain'), global_step=self.global_step)

    def restore_model(self, sess: tf.Session, model_path: str):
        """
        Restore model.
        :param sess: Tensorflow session
        :param model_path: path to model
        """
        latest_checkpoint = tf.train.latest_checkpoint(os.path.join(model_path, 'model'))
        self.saver.restore(sess, latest_checkpoint)


def load_dataset(dataset_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load dataset for pretrain.
    :param dataset_path: path to dataset (.npz file)
    :return: train_input, train_status, train_rewards, test_input, test_status, test_rewards
    """
    dataset = np.load(dataset_path)
    network_input = dataset["network_input"]
    status = dataset["status"]
    rewards = dataset["rewards"]
    train_idx = dataset["train_idx"]
    test_idx = dataset["test_idx"]

    train_input = network_input[train_idx, :]
    test_input = network_input[test_idx, :]
    train_status = status[train_idx]
    test_status = status[test_idx]
    train_rewards = np.reshape(rewards[train_idx], (-1, 1))
    test_rewards = np.reshape(rewards[test_idx], (-1, 1))
    # convert to one-hot encoding
    train_status = to_categorical(train_status)
    test_status = to_categorical(test_status)
    return train_input, train_status, train_rewards, test_input, test_status, test_rewards


def get_args():
    parser = argparse.ArgumentParser(description="Train pretrained reward network")

    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--save-interval', type=int, default=10)
    parser.add_argument("--transition-network", nargs="+", default=[200, 200],
                        help='how many neurons in each layer of transition network')
    parser.add_argument("--reward-network", nargs="+", default=[300, 300],
                        help='how many neurons in each layer of reward network')
    parser.add_argument('--data-path', type=str,
                        default=f"{ROOT_STR}/data/supervised_data/data.npz",
                        help='path to dataset')
    parser.add_argument('--save-path', type=str,
                        default=f"{ROOT_STR}/pretrain_network",
                        help='where to save the model')
    return parser.parse_args()


def main(args):
    model_name = datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S')
    os.makedirs(os.path.join(args.save_path, model_name))
    # Save hyperparameters
    parameters = dict(vars(args))
    with open(os.path.join(args.save_path, model_name, "config.yml"), "w") as f:
        yaml.dump(parameters, f, default_flow_style=False)

    # Load dataset
    train_input, train_status, train_rewards, test_input, test_status, test_rewards = load_dataset(args.data_path)
    # Initialize model
    network = PretrainNetwork(transition_network=args.transition_network,
                              reward_network=args.reward_network)

    # Begin training
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(
            os.path.join(args.save_path, model_name, 'tensorboard', 'train'),
            sess.graph)
        test_writer = tf.summary.FileWriter(
            os.path.join(args.save_path, model_name, 'tensorboard', 'test'),
            sess.graph)
        sess.run(tf.global_variables_initializer())

        n_batches_train = int(len(train_input) / args.batch_size)
        n_batched_test = int(len(test_input) / args.batch_size)
        print('Training...')
        for i in range(args.num_epochs):
            total_status_loss_train = 0
            total_reward_loss_train = 0
            for batch in range(n_batches_train):
                # Train for one batch
                idx_start = batch * args.batch_size
                idx_end = min((batch + 1) * args.batch_size, len(train_input))
                s_loss, r_loss, train_summary, current_global_step = network.train(sess,
                                                                                   train_input[idx_start:idx_end],
                                                                                   train_status[idx_start:idx_end],
                                                                                   train_rewards[idx_start:idx_end])
                total_status_loss_train += s_loss
                total_reward_loss_train += r_loss
                train_writer.add_summary(train_summary, current_global_step)

            total_status_loss_test = 0
            total_reward_loss_test = 0
            for batch in range(n_batched_test):
                # Test for one batch
                idx_start = batch * args.batch_size
                idx_end = min((batch + 1) * args.batch_size, len(test_input))
                s_loss, r_loss, test_summary = network.test(sess,
                                                            test_input[idx_start:idx_end],
                                                            test_status[idx_start:idx_end],
                                                            test_rewards[idx_start:idx_end])
                total_status_loss_test += s_loss
                total_reward_loss_test += r_loss
                test_writer.add_summary(test_summary, current_global_step)
            print(
                f"Iter: {i}, Train status loss: {total_status_loss_train / n_batches_train:.4f}, "
                f"Train reward loss: {total_reward_loss_train / n_batches_train:.4f}, "
                f"Test status loss: {total_status_loss_test / n_batched_test:.4f}, "
                f"Test reward loss: {total_reward_loss_test / n_batched_test:.4f}")
            if (i + 1) % args.save_interval == 0:
                network.save_model(sess, model_name, args.save_path)
                print(f"Save check point to {args.save_path}/{model_name}")
            train_writer.flush()
            test_writer.flush()


if __name__ == '__main__':
    args = get_args()
    main(args)
