import tensorflow as tf
import numpy as np
import os
import random

from config import parse_args
from module import model
from data import create_dataset

class Network(object):
    def __init__(self, args):
        self.args = args

        CKPT_DIR = args.ckpt_dir

        # Create ckpt directory if it does not exist
        if not os.path.exists(CKPT_DIR):
            os.mkdir(CKPT_DIR)

        # Print Arguments
        args = args.__dict__
        print("Arguments : ")

        for key, value in sorted(args.items()):
            print('\t%15s:\t%s' % (key, value))

    def build(self):

        # Parameters
        CTX_UP = self.args.ctx_up
        CTX_LEFT = self.args.ctx_left
        CTX_TOTAL = (CTX_LEFT * 2 + 1) * CTX_UP + CTX_LEFT

        LAYER_NUM = self.args.layer_num
        HIDDEN_UNIT = self.args.hidden_unit

        LAMBDA_CTX = self.args.lambda_ctx

        LR = self.args.lr

        self.input = tf.placeholder(tf.float32, (None, 3 * CTX_TOTAL - 3))
        self.output = tf.placeholder(tf.float32, (None, 3))

        y_gt = tf.slice(self.output, [0, 0], [-1, 1])
        u_gt = tf.slice(self.output, [0, 1], [-1, 1])
        v_gt = tf.slice(self.output, [0, 2], [-1, 1])

        out_y, hidden_y = model(self.input, LAYER_NUM, HIDDEN_UNIT, 'pred_y')

        input_f2 = tf.concat([hidden_y, self.input, y_gt, tf.expand_dims(out_y[:,0], axis=1)], axis=1)

        out_u, hidden_u = model(input_f2, LAYER_NUM, HIDDEN_UNIT, 'pred_u')

        input_f3 = tf.concat([hidden_u, self.input, y_gt, tf.expand_dims(out_y[:, 0], axis=1), u_gt, tf.expand_dims(out_u[:, 0], axis=1)], axis=1)

        out_v, _, = model(input_f3, LAYER_NUM, HIDDEN_UNIT, 'pred_v')

        pred_y = out_y[:, 0]
        pred_u = out_u[:, 0]
        pred_v = out_v[:, 0]
        ctx_y  = tf.nn.relu(out_y[:, 1])
        ctx_u  = tf.nn.relu(out_u[:, 1])
        ctx_v  = tf.nn.relu(out_v[:, 1])

        predError_y = abs(tf.subtract(pred_y, tf.squeeze(y_gt, axis=1)))
        predError_u = abs(tf.subtract(pred_u, tf.squeeze(u_gt, axis=1)))
        predError_v = abs(tf.subtract(pred_v, tf.squeeze(v_gt, axis=1)))

        loss_pred_y = tf.reduce_mean(predError_y)
        loss_pred_u = tf.reduce_mean(predError_u)
        loss_pred_v = tf.reduce_mean(predError_v)

        loss_ctx_y = LAMBDA_CTX * tf.reduce_mean(abs(tf.subtract(ctx_y, predError_y)))
        loss_ctx_u = LAMBDA_CTX * tf.reduce_mean(abs(tf.subtract(ctx_u, predError_u)))
        loss_ctx_v = LAMBDA_CTX * tf.reduce_mean(abs(tf.subtract(ctx_v, predError_v)))

        loss_y = loss_pred_y + loss_ctx_y
        loss_u = loss_pred_u + loss_ctx_u
        loss_v = loss_pred_v + loss_ctx_v

        loss_yuv = (loss_pred_y + loss_pred_u + loss_pred_v) + (loss_ctx_y + loss_ctx_u + loss_ctx_v)

        t_vars = tf.trainable_variables()
        y_vars = [var for var in t_vars if 'pred_y' in var.name]
        u_vars = [var for var in t_vars if 'pred_u' in var.name]
        v_vars = [var for var in t_vars if 'pred_v' in var.name]

        self.optimizer_y = tf.train.AdamOptimizer(LR).minimize(loss_y, var_list=y_vars)
        self.optimizer_u = tf.train.AdamOptimizer(LR).minimize(loss_u, var_list=u_vars)
        self.optimizer_v = tf.train.AdamOptimizer(LR).minimize(loss_v, var_list=v_vars)
        self.optimizer_yuv = tf.train.AdamOptimizer(LR).minimize(loss_yuv, var_list=t_vars)

        # Variables
        self.loss_y = loss_y
        self.loss_u = loss_u
        self.loss_v = loss_v
        self.loss_yuv = loss_yuv
        self.loss_pred_y = loss_pred_y
        self.loss_pred_u = loss_pred_u
        self.loss_pred_v = loss_pred_v
        self.loss_pred_yuv = loss_pred_v + loss_pred_u + loss_pred_v
        self.loss_ctx_y = loss_ctx_y
        self.loss_ctx_u = loss_ctx_u
        self.loss_ctx_v = loss_ctx_v
        self.loss_ctx_yuv = loss_ctx_y + loss_ctx_u + loss_ctx_v
        self.ctx_y = ctx_y
        self.ctx_u = ctx_u
        self.ctx_v = ctx_v

    def train(self):

        GPU_NUM = self.args.gpu_num
        CKPT_DIR = self.args.ckpt_dir
        LOAD = self.args.load
        EPOCH = self.args.epoch
        BATCH_SIZE = self.args.batch_size
        SAVE_INTERVAL = self.args.save_interval

        # Assign GPU
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_NUM)

        self.check_data_exist()

        global_step = tf.Variable(0, trainable=False)
        increase = tf.assign_add(global_step, 1)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=1)
            ckpt = tf.train.get_checkpoint_state(CKPT_DIR)

            # Load model if trained before
            if ckpt and LOAD:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("Model Loaded")

            # Load dataset
            Xdata_train, Ydata_train, data_num = self.load_dataset("train")
            Xdata_valid, Ydata_valid, _ = self.load_dataset("valid")

            epoch = sess.run(global_step)
            EPOCH = EPOCH + EPOCH % 4

            while epoch < EPOCH:
                sess.run(increase)

                loss_pred_epoch_y = loss_pred_epoch_u = loss_pred_epoch_v = 0
                loss_ctx_epoch_y = loss_ctx_epoch_u = loss_ctx_epoch_v = 0

                step = 0

                if epoch == 0:
                    print("========== Train Y Channel ==========")
                    optimizer = self.optimizer_y
                elif epoch == EPOCH / 4:
                    print("========== Train U Channel ==========")
                    optimizer = self.optimizer_u
                elif epoch == 2 * EPOCH / 4:
                    print("========== Train V Channel ==========")
                    optimizer = self.optimizer_v
                elif epoch == 3 * EPOCH / 4:
                    print("========== Train YUV Channel ==========")
                    optimizer = self.optimizer_yuv
                else:
                    pass

                for step in range(int(data_num / BATCH_SIZE) + 1):
                    batch_input, batch_label = self.get_batch(Xdata_train, Ydata_train, step, BATCH_SIZE)

                    if len(batch_input) != 0:
                        feed_dict = {
                            self.input : batch_input,
                            self.output: batch_label
                        }

                        sess.run(optimizer, feed_dict=feed_dict)
                        loss_p_y, loss_p_u, loss_p_v, loss_c_y, loss_c_u, loss_c_v = sess.run(
                            [self.loss_pred_y, self.loss_pred_u, self.loss_pred_v, self.loss_ctx_y, self.loss_ctx_u, self.loss_ctx_v], feed_dict=feed_dict)

                        loss_pred_epoch_y += loss_p_y
                        loss_pred_epoch_u += loss_p_u
                        loss_pred_epoch_v += loss_p_v

                        loss_ctx_epoch_y += loss_c_y
                        loss_ctx_epoch_u += loss_c_u
                        loss_ctx_epoch_v += loss_c_v

                loss_pred_epoch_y /= step
                loss_pred_epoch_u /= step
                loss_pred_epoch_v /= step

                loss_ctx_epoch_y /= step
                loss_ctx_epoch_u /= step
                loss_ctx_epoch_v /= step

                loss_epoch_y = loss_pred_epoch_y + loss_ctx_epoch_y
                loss_epoch_u = loss_pred_epoch_u + loss_ctx_epoch_u
                loss_epoch_v = loss_pred_epoch_v + loss_ctx_epoch_v

                print('%04d\n' % (epoch + 1),
                      '***Y***   lossPred=', '{:9.4f}'.format(loss_pred_epoch_y), 'lossContext=', '{:9.4f}'.format(loss_ctx_epoch_y), 'Loss=', '{:9.4f}\n'.format(loss_epoch_y),
                      '***U***   lossPred=', '{:9.4f}'.format(loss_pred_epoch_u), 'lossContext=', '{:9.4f}'.format(loss_ctx_epoch_u), 'Loss=', '{:9.4f}\n'.format(loss_epoch_u),
                      '***V***   lossPred=', '{:9.4f}'.format(loss_pred_epoch_v), 'lossContext=', '{:9.4f}'.format(loss_ctx_epoch_v), 'Loss=', '{:9.4f}\n'.format(loss_epoch_v),
                      '***YUV*** lossPred=', '{:9.4f}'.format(loss_pred_epoch_y + loss_pred_epoch_u + loss_pred_epoch_v), 'lossContext=',
                      '{:9.4f}'.format(loss_ctx_epoch_y + loss_ctx_epoch_u + loss_ctx_epoch_v), 'Loss=', '{:9.4f}'.format(loss_epoch_y + loss_epoch_u + loss_epoch_v))

                if (epoch + 1) % SAVE_INTERVAL == 0:
                    saver.save(sess, CKPT_DIR + 'model_', global_step=epoch + 1)
                    self.print_weights('y')
                    self.print_weights('u')
                    self.print_weights('v')
                    print("Model Saved")

                epoch = sess.run(global_step)

            self.print_all_weights()

    def check_data_exist(self):
        path = self.args.data_dir + "npy/"
        filelist = os.listdir(path)

        if not (len(filelist) == 6):
            create_dataset(self.args, "train")
            create_dataset(self.args, "valid")
            create_dataset(self.args, "test")

    def load_dataset(self, data_type, shuffle=True):
        path = self.args.data_dir + 'npy/'

        Xdata = np.load(path + 'Xdata_' + data_type + '.npy')
        Ydata = np.load(path + 'Ydata_' + data_type + '.npy')

        data_num = Xdata.shape[0]

        # Shuffle dataset
        if shuffle:
            data_idx = list(range(data_num))
            random.shuffle(data_idx)

            Xdata = Xdata[data_idx]
            Ydata = Ydata[data_idx]

        return Xdata, Ydata, data_num

    @staticmethod
    def get_batch(Xdata, Ydata, step, batch_size):
        offset = step * batch_size

        if step == int(len(Xdata) / batch_size):
            batch_input = Xdata[offset:]
            batch_label = Ydata[offset:]
        else:
            batch_input = Xdata[offset:(offset + batch_size)]
            batch_label = Ydata[offset:(offset + batch_size)]

        return batch_input, batch_label

    def print_weights(self, channel='y'):

        HIDDEN_UNIT = self.args.hidden_unit
        CTX_UP = self.args.ctx_up
        CTX_LEFT = self.args.ctx_left

        W = [v for v in tf.trainable_variables() if (('kernel' in v.name) and ('pred_' + channel in v.name))]
        b = [v for v in tf.trainable_variables() if (('bias' in v.name) and ('pred_' + channel in v.name))]

        n_layer = len(W)

        W_ = []
        b_ = []

        for i in range(n_layer):
            W_.append(W[i].eval())
            b_.append(b[i].eval())

        n_in = W_[0].shape[0]
        n_hidden = HIDDEN_UNIT
        n_out = W[-1].shape[1]

        filename = channel + '_weights.txt'

        f = open(filename, 'w')

        f.write(str(n_in) + '\n')
        f.write(str(n_hidden) + '\n')
        f.write(str(n_out) + '\n')
        f.write(str(n_layer) + '\n')
        f.write(str(CTX_UP) + '\n')
        f.write(str(CTX_LEFT) + '\n')

        for k in range(n_layer):
            for i in range(W_[k].shape[0]):
                for j in range(W_[k].shape[1]):
                    f.write(str(W_[k][i,j]) + '\t')
                f.write('\n')

        for k in range(n_layer):
            for j in range(b_[k].shape[0]):
                f.write(str(b_[k][j]) + '\t')
            f.write('\n')


        f.close()

        filename = os.path.abspath("../") + "/c_compression/x64/Release/" + channel + '_weights.txt'

        f = open(filename, 'w')

        f.write(str(n_in) + '\n')
        f.write(str(n_hidden) + '\n')
        f.write(str(n_out) + '\n')
        f.write(str(n_layer) + '\n')
        f.write(str(CTX_UP) + '\n')
        f.write(str(CTX_LEFT) + '\n')

        for k in range(n_layer):
            for i in range(W_[k].shape[0]):
                for j in range(W_[k].shape[1]):
                    f.write(str(W_[k][i,j]) + '\t')
                f.write('\n')

        for k in range(n_layer):
            for j in range(b_[k].shape[0]):
                f.write(str(b_[k][j]) + '\t')
            f.write('\n')


        f.close()


    def print_all_weights(self):

        GPU_NUM = self.args.gpu_num
        CKPT_DIR = self.args.ckpt_dir

        # Assign GPU
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_NUM)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=1)
            ckpt = tf.train.get_checkpoint_state(CKPT_DIR)

            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Model Loaded")

            self.print_weights('y')
            self.print_weights('u')
            self.print_weights('v')


if __name__ == "__main__":

    args = parse_args()
    my_net = Network(args)
    my_net.build()
    my_net.train()
    my_net.print_all_weights()