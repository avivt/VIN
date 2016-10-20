# VI network using THEANO, takes batches of state input
from NNobj import *
from theano_utils import *


class fcn(NNobj):
    "Class for a fully connected convolutional network"
    def __init__(self, model="FCN", im_size=[28, 28], dropout=False, devtype="cpu", grad_check=False, reg=0,
                 statebatchsize=10, batchsize=128):
        self.im_size = im_size                # input image size
        self.model = model
        self.reg = reg                        # regularization (currently not implemented)
        self.batchsize = batchsize            # batch size for training
        self.statebatchsize = statebatchsize  # number of state inputs for every image input, since each image is the
        # same for many states in the data
        np.random.seed(0)
        print(model)
        theano.config.blas.ldflags = "-L/usr/local/lib -lopenblas"

        # X input : l=2 stacked images: obstacle map and reward function prior
        self.X = T.ftensor4(name="X")
        # S1,S2 input : state position (vertical and horizontal position)
        self.S1 = T.bmatrix("S1")  # state first dimension * statebatchsize
        self.S2 = T.bmatrix("S2")  # state second dimension * statebatchsize
        self.y = T.bvector("y")    # output action * statebatchsize

        l = 2
        l_1 = 150   # channels (filters) in first conv layer
        l_2 = 150
        l_3 = 10

        self.fcn_net = FCN(in_x=self.X, in_s1=self.S1, in_s2=self.S2, in_x_channels=l, imsize=self.im_size,
                           batchsize=self.batchsize, state_batch_size=self.statebatchsize, l_1=l_1, l_2=l_2,
                           l_3=l_3)
        self.p_of_y = self.fcn_net.output
        self.params = self.fcn_net.params
        # Total 1910 parameters

        self.cost = -T.mean(T.log(self.p_of_y)[T.arange(self.y.shape[0]),
                                               self.y], dtype=theano.config.floatX)
        self.y_pred = T.argmax(self.p_of_y, axis=1)
        self.err = T.mean(T.neq(self.y_pred, self.y.flatten()), dtype=theano.config.floatX)

        self.computeloss = theano.function(inputs=[self.X, self.S1, self.S2, self.y],
                                           outputs=[self.err, self.cost])
        self.y_out = theano.function(inputs=[self.X, self.S1, self.S2], outputs=[self.y_pred])

    def run_training(self, input, stepsize=0.01, epochs=10, output='None', batch_size=128, grad_check=True,
                     profile=False, data_fraction=1):
        # run training from input matlab data file, and save test data prediction in output file
        # load data from Matlab file, including
        # im_data: flattened images
        # state_data: concatenated one-hot vectors for each state variable
        # state_xy_data: state variable (x,y position)
        # label_data: one-hot vector for action (state difference)
        matlab_data = sio.loadmat(input)
        im_data = matlab_data["batch_im_data"]
        im_data = (im_data - 1)/255  # obstacles = 1, free zone = 0
        value_data = matlab_data["batch_value_data"]
        state1_data = matlab_data["state_x_data"]
        state2_data = matlab_data["state_y_data"]
        label_data = matlab_data["batch_label_data"]
        ydata = label_data.astype('int8')
        Xim_data = im_data.astype(theano.config.floatX)
        Xim_data = Xim_data.reshape(-1, 1, self.im_size[0], self.im_size[1])
        Xval_data = value_data.astype(theano.config.floatX)
        Xval_data = Xval_data.reshape(-1, 1, self.im_size[0], self.im_size[1])
        Xdata = np.append(Xim_data, Xval_data, axis=1)
        S1data = state1_data.astype('int8')
        S2data = state2_data.astype('int8')

        all_training_samples = int(6/7.0*Xdata.shape[0])
        training_samples = int(data_fraction * all_training_samples)
        Xtrain = Xdata[0:training_samples]
        S1train = S1data[0:training_samples]
        S2train = S2data[0:training_samples]
        ytrain = ydata[0:training_samples]

        Xtest = Xdata[all_training_samples:]
        S1test = S1data[all_training_samples:]
        S2test = S2data[all_training_samples:]
        ytest = ydata[all_training_samples:]
        ytest = ytest.flatten()

        sortinds = np.random.permutation(training_samples)
        Xtrain = Xtrain[sortinds]
        S1train = S1train[sortinds]
        S2train = S2train[sortinds]
        ytrain = ytrain[sortinds]
        ytrain = ytrain.flatten()

        self.updates = rmsprop_updates_T(self.cost, self.params, stepsize=stepsize)
        self.train = theano.function(inputs=[self.X, self.S1, self.S2, self.y], outputs=[], updates=self.updates)

        print fmt_row(10, ["Epoch", "Train NLL", "Train Err", "Test NLL", "Test Err", "Epoch Time"])
        for i_epoch in xrange(int(epochs)):
            tstart = time.time()
            # do training
            for start in xrange(0, Xtrain.shape[0], batch_size):
                end = start+batch_size
                if end <= Xtrain.shape[0]:
                    self.train(Xtrain[start:end], S1train[start:end], S2train[start:end],
                               ytrain[start*self.statebatchsize:end*self.statebatchsize])
            elapsed = time.time() - tstart
            # compute losses
            trainerr = 0.
            trainloss = 0.
            testerr = 0.
            testloss = 0.
            num = 0
            for start in xrange(0, Xtest.shape[0], batch_size):
                end = start+batch_size
                if end <= Xtest.shape[0]:
                    num += 1
                    trainerr_, trainloss_ = self.computeloss(Xtrain[start:end], S1train[start:end], S2train[start:end],
                                                             ytrain[start*self.statebatchsize:end*self.statebatchsize])
                    testerr_, testloss_ = self.computeloss(Xtest[start:end], S1test[start:end], S2test[start:end],
                                                           ytest[start*self.statebatchsize:end*self.statebatchsize])
                    trainerr += trainerr_
                    trainloss += trainloss_
                    testerr += testerr_
                    testloss += testloss_
            print fmt_row(10, [i_epoch, trainloss/num, trainerr/num, testloss/num, testerr/num, elapsed])

    def predict(self, input):
        # NN output for a single input, read from file
        matlab_data = sio.loadmat(input)
        im_data = matlab_data["im_data"]
        im_data = (im_data - 1)/255  # obstacles = 1, free zone = 0
        state_data = matlab_data["state_xy_data"]
        value_data = matlab_data["value_data"]
        xim_test = im_data.astype(theano.config.floatX)
        xim_test = xim_test.reshape(-1, 1, self.im_size[0], self.im_size[1])
        xval_test = value_data.astype(theano.config.floatX)
        xval_test = xval_test.reshape(-1, 1, self.im_size[0], self.im_size[1])
        x_test = np.append(xim_test, xval_test, axis=1)
        s_test = state_data.astype('int8')
        s1_test = s_test[:, 0].reshape([1, 1])
        s2_test = s_test[:, 1].reshape([1, 1])
        out = self.y_out(x_test, s1_test, s2_test)
        return out[0][0]

    def load_weights(self, infile="weight_dump.pk"):
        dump = pickle.load(open(infile, 'r'))
        [n.set_value(p) for n, p in zip(self.params, dump)]

    def save_weights(self, outfile="weight_dump.pk"):
        pickle.dump([n.get_value() for n in self.params], open(outfile, 'w'))


class FCN(object):
    """FCN network"""
    def __init__(self, in_x, in_s1, in_s2, in_x_channels, imsize, batchsize=128,
                 state_batch_size=1, l_1=150, l_2=150, l_3=150):
        """
        Allocate a FCN network with shared variable internal parameters. Assumes 16X16 images

        :type in_x: theano.tensor.dtensor4
        :param in_x: symbolic input image tensor, of shape [batchsize, in_x_channels, imsize[0], imsize[1]]
        Typically : first channel is image, second is the reward prior.

        :type in_s1: theano.tensor.bmatrix
        :param in_s1: symbolic input batches of vertical positions, of shape [batchsize, state_batch_size]

        :type in_s2: theano.tensor.bmatrix
        :param in_s2: symbolic input batches of horizontal positions, of shape [batchsize, state_batch_size]

        :type in_x_channels: int32
        :param in_x_channels: number of input channels

        :type imsize: tuple or list of length 2
        :param imsize: (image height, image width)

        :type batchsize: int32
        :param batchsize: batch size

        :type state_batch_size: int32
        :param state_batch_size: number of state inputs for each sample

        :type l_1: int32
        :param l_1: number of filters in first conv layer

        :type l_2: int32
        :param l_2: number of filters in second conv layer

        :type l_3: int32
        :param l_3: number of filters in third conv layer

        """
        self.b1 = theano.shared((np.random.randn(l_1) * 0.01).astype(theano.config.floatX))
        self.w1 = init_weights_T(l_1, in_x_channels, imsize[0]*2-1, imsize[1]*2-1)
        self.h1 = T.nnet.conv2d(in_x, self.w1, input_shape=[batchsize, self.w1.shape.eval()[1], imsize[0], imsize[1]],
                                border_mode=(imsize[0]-1, imsize[1]-1),
                                filter_shape=[l_1, in_x_channels, imsize[0]*2-1, imsize[1]*2-1])
        self.h1 = T.nnet.relu(self.h1 + self.b1.dimshuffle('x', 0, 'x', 'x'))

        self.w2 = init_weights_T(l_2, l_1, 1, 1)
        self.h2 = conv2D_keep_shape(self.h1, self.w2, image_shape=[batchsize, self.w1.shape.eval()[0],
                                                               imsize[0], imsize[1]],
                                filter_shape=[l_2, l_1, 1, 1])
        self.b2 = theano.shared((np.random.randn(l_2) * 0.01).astype(theano.config.floatX))  # 150 parameters
        self.h2 = T.nnet.relu(self.h2 + self.b2.dimshuffle('x', 0, 'x', 'x'))

        self.w3 = init_weights_T(l_3, l_2, 1, 1)
        self.h3 = conv2D_keep_shape(self.h2, self.w3, image_shape=[batchsize, self.w2.shape.eval()[0],
                                                               imsize[0], imsize[1]],
                                filter_shape=[l_3, l_2, 1, 1])
        self.b3 = theano.shared((np.random.randn(l_3) * 0.01).astype(theano.config.floatX))  # 150 parameters
        self.h3 = T.nnet.relu(self.h3 + self.b3.dimshuffle('x', 0, 'x', 'x'))

        # Select the conv-net channels at the state position (S1,S2). This is the FCN thing.
        self.h_out = self.h3[T.extra_ops.repeat(T.arange(self.h3.shape[0]), state_batch_size), :, in_s1.flatten(),
                             in_s2.flatten()]

        # softmax output weights
        self.w_o = init_weights_T(l_3, 8)
        self.output = T.nnet.softmax(T.dot(self.h_out, self.w_o))

        self.params = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, self.w_o]
