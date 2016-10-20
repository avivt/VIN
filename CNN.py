# VI network using THEANO, takes batches of state input
from NNobj import *
from theano_utils import *


class cnn(NNobj):
    "Class for a convolutional neural network, inthe style of LeNet/Alexnet"
    def __init__(self, model="CNN", im_size=[28, 28], dropout=False, devtype="cpu", grad_check=False, reg=0,
                 batchsize=128):
        self.im_size = im_size                # input image size
        self.model = model
        self.reg = reg                        # regularization (currently not implemented)
        self.batchsize = batchsize            # batch size for training
        np.random.seed(0)
        print(model)
        # theano.config.blas.ldflags = "-L/usr/local/lib -lopenblas"

        # X input : l=3 stacked images: obstacle map, goal map, current state map
        self.X = T.ftensor4(name="X")
        self.y = T.bvector("y")    # output action

        l = 3
        filter_sizes = [[50, 3, 3],
                        [50, 3, 3],
                        [100, 3, 3],
                        [100, 3, 3],
                        [100, 3, 3]]
        poolings = [2, 1, 2, 1, 1]

        self.cnn_net = CNN(in_x=self.X, in_x_channels=l, imsize=self.im_size,
                           batchsize=self.batchsize, filter_sizes=filter_sizes,
                           poolings=poolings)
        self.p_of_y = self.cnn_net.output
        self.params = self.cnn_net.params
        # Total 1910 parameters

        self.cost = -T.mean(T.log(self.p_of_y)[T.arange(self.y.shape[0]),
                                               self.y], dtype=theano.config.floatX)
        self.y_pred = T.argmax(self.p_of_y, axis=1)
        self.err = T.mean(T.neq(self.y_pred, self.y.flatten()), dtype=theano.config.floatX)

        self.computeloss = theano.function(inputs=[self.X, self.y],
                                           outputs=[self.err, self.cost])
        self.y_out = theano.function(inputs=[self.X], outputs=[self.y_pred])
        self.updates = []
        self.train = []

    def run_training(self, input, stepsize=0.01, epochs=10, output='None', batch_size=128, grad_check=True,
                     profile=False, data_fraction=1):
        # run training from input matlab data file, and save test data prediction in output file
        # load data from Matlab file, including
        # im_data: flattened images
        # value_data: flattened reward image
        # state_data: flattened state images
        # label_data: one-hot vector for action (state difference)
        matlab_data = sio.loadmat(input)
        im_data = matlab_data["im_data"]
        im_data = (im_data - 1)/255  # obstacles = 1, free zone = 0
        value_data = matlab_data["value_data"]
        state1_data = matlab_data["state_x_data"]
        state2_data = matlab_data["state_y_data"]
        label_data = matlab_data["label_data"]
        y_data = label_data.astype('int8')
        x_im_data = im_data.astype(theano.config.floatX)
        x_im_data = x_im_data.reshape(-1, 1, self.im_size[0], self.im_size[1])
        x_val_data = value_data.astype(theano.config.floatX)
        x_val_data = x_val_data.reshape(-1, 1, self.im_size[0], self.im_size[1])
        x_state_data = np.zeros_like(x_im_data)
        for i in x_state_data.shape[0]:
            pos1 = state1_data[i]
            pos2 = state2_data[i]
            x_state_data[i, 0, pos1, pos2] = 1
        x_data = np.append(x_im_data, x_val_data, axis=1)
        x_data = np.append(x_data, x_state_data, axis=1)

        all_training_samples = int(6/7.0*x_data.shape[0])
        training_samples = int(data_fraction * all_training_samples)
        x_train = x_data[0:training_samples]
        y_train = y_data[0:training_samples]

        x_test = x_data[all_training_samples:]
        y_test = y_data[all_training_samples:]
        y_test = y_test.flatten()

        sortinds = np.random.permutation(training_samples)
        x_train = x_train[sortinds]
        y_train = y_train[sortinds]
        y_train = y_train.flatten()

        self.updates = rmsprop_updates_T(self.cost, self.params, stepsize=stepsize)
        self.train = theano.function(inputs=[self.X, self.y], outputs=[], updates=self.updates)

        print fmt_row(10, ["Epoch", "Train NLL", "Train Err", "Test NLL", "Test Err", "Epoch Time"])
        for i_epoch in xrange(int(epochs)):
            tstart = time.time()
            # do training
            for start in xrange(0, x_train.shape[0], batch_size):
                end = start+batch_size
                if end <= x_train.shape[0]:
                    self.train(x_train[start:end], y_train[start:end])
            elapsed = time.time() - tstart
            # compute losses
            trainerr = 0.
            trainloss = 0.
            testerr = 0.
            testloss = 0.
            num = 0
            for start in xrange(0, x_test.shape[0], batch_size):
                end = start+batch_size
                if end <= x_test.shape[0]:
                    num += 1
                    trainerr_, trainloss_ = self.computeloss(x_train[start:end], y_train[start:end])
                    testerr_, testloss_ = self.computeloss(x_test[start:end], y_test[start:end])
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
        # state_data = matlab_data["state_data"]
        state_data = matlab_data["state_xy_data"]
        value_data = matlab_data["value_data"]
        x_im_test = im_data.astype(theano.config.floatX)
        x_im_test = x_im_test.reshape(-1, 1, self.im_size[0], self.im_size[1])
        x_val_test = value_data.astype(theano.config.floatX)
        x_val_test = x_val_test.reshape(-1, 1, self.im_size[0], self.im_size[1])
        x_state_test = np.zeros_like(x_im_test)
        x_state_test[0, 0, state_data[0, 0], state_data[0, 1]] = 1
        x_test = np.append(x_im_test, x_val_test, axis=1)
        x_test = np.append(x_test, x_state_test, axis=1)
        out = self.y_out(x_test)
        return out[0][0]

    def load_weights(self, infile="weight_dump.pk"):
        dump = pickle.load(open(infile, 'r'))
        [n.set_value(p) for n, p in zip(self.params, dump)]

    def save_weights(self, outfile="weight_dump.pk"):
        pickle.dump([n.get_value() for n in self.params], open(outfile, 'w'))


class CNN(object):
    """CNN network"""
    def __init__(self, in_x, in_x_channels, imsize, batchsize=128,
                 filter_sizes=[[50, 3, 3], [100, 3, 3]], poolings=[2, 2]):
        """
        Allocate a CNN network with shared variable internal parameters.

        :type in_x: theano.tensor.dtensor4
        :param in_x: symbolic input image tensor, of shape [batchsize, in_x_channels, imsize[0], imsize[1]]
        Typically : first channel is image, second is the reward prior, third is the current state image.

        :type in_x_channels: int32
        :param in_x_channels: number of input channels

        :type imsize: tuple or list of length 2
        :param imsize: (image height, image width)

        :type batchsize: int32
        :param batchsize: batch size

        :type filter_sizes: int32 list of int32 3-tuples
        :param filter_sizes: list of filter sizes for each layer, each a list of 3 integers:
        num_filters,filter_width,filter_height

        :type batchsize: int32 list
        :param batchsize: list of pooling ratios after each layer (assumed symmetric)
        """
        assert len(filter_sizes) == len(poolings)
        n_conv_layers = len(filter_sizes)
        self.params = []
        # first conv layer
        prev_layer = ConvLayer(in_x, filter_shape=[filter_sizes[0][0], in_x_channels, filter_sizes[0][1],
                                                   filter_sizes[0][2]],
                               image_shape=[batchsize, in_x_channels, imsize[0], imsize[1]],
                               poolsize=(poolings[0], poolings[0]))
        self.params = self.params + prev_layer.params
        # then the rest of the conv layers
        for l in range(1, n_conv_layers):
            new_layer = ConvLayer(prev_layer.output,
                                  filter_shape=[filter_sizes[l][0], prev_layer.out_shape[1], filter_sizes[l][1],
                                                filter_sizes[l][2]],
                                  image_shape=prev_layer.out_shape,
                                  poolsize=(poolings[l], poolings[l]))
            self.params = self.params + new_layer.params
            prev_layer = new_layer
        # fully connected layer
        final_conv_shape = new_layer.out_shape
        flat_conv_out = new_layer.output.flatten(ndim=2)
        flat_shape = [final_conv_shape[0], final_conv_shape[1]*final_conv_shape[2]*final_conv_shape[3]]
        self.w_o = init_weights_T(flat_shape[1], 8)
        self.output = T.nnet.softmax(T.dot(flat_conv_out, self.w_o))
        self.params = self.params + [self.w_o]
