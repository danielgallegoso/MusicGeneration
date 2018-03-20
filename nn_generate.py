import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
from utils import generate_dataset_iterator, save_decoding
import pickle
from nn import initialize_nn

# Network Parameters
num_input = 128*2 + 100
timesteps = 100
num_hidden = 512 # hidden layer num of features
beta = .003
epochs = 100
threshold = .5
filename = 'out'
prob_1 = 0.0087827912566791136


def main():
    # with open('model0.p'.format(epoch)) as f:
    #     model = pickle.load(f)
    #     sess = model['sess']
    #     loss_op = model['loss_op']
    #     train_op = model['train_op']
    #     X = model['X']
    #     Y = model['Y']
    #     init = model['init']
    #     prediction = model['prediction']
    init, train_op, X, Y, loss_op, prediction, saver = initialize_nn()

    with tf.Session() as sess:
        saver.restore(sess, 'models/model0.ckpt')
        fake = None
        for decoding in generate_dataset_iterator('dev'):
            fake = decoding
            break
        decoding = fake[:timesteps]
        for i in range(timesteps):
            batch_x = np.array([np.copy(decoding)])
            batch_y = np.array([np.concatenate([np.zeros((1,num_input)), decoding[1:]], axis=0)])
            pred = sess.run(prediction, feed_dict={X: batch_x, Y: batch_y})
            newFrame = pred[0][-1]
            # print newFrame
            on = np.argmax(newFrame[:128])
            off = np.argmax(newFrame[128:2*128])
            shift = np.argmax(newFrame[2*128:])
            newFrame = np.zeros(2*128 + 100)
            newFrame[on] = 1
            newFrame[128+off] = 1
            newFrame[2*128 + shift] = 1
            # newFrame[newFrame > threshold] = 1
            # newFrame[newFrame <= threshold] = 0
            decoding = np.concatenate([decoding[1:], [newFrame]], axis=0)
        save_decoding(decoding, 'goat.mid')


if __name__== "__main__":
  main()
