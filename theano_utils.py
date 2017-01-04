# THEANO NN utils
import numpy as np
import theano
import theano.tensor as T


def init_weights_T(*shape):
    return theano.shared((np.random.randn(*shape) * 0.01).astype(theano.config.floatX))


def conv2D_keep_shape(x, w, image_shape, filter_shape, subsample=(1, 1)):
    # crop output to same size as input
    fs = T.shape(w)[2] - 1  # this is the filter size minus 1
    ims = T.shape(x)[2]     # this is the image size
    return theano.sandbox.cuda.dnn.dnn_conv(img=x,
                                            kerns=w,
                                            border_mode='full',
                                            subsample=subsample,
                                            )[:, :, fs/2:ims+fs/2, fs/2:ims+fs/2]


def rmsprop_updates_T(cost, params, stepsize=0.001, rho=0.9, epsilon=1e-6):
    # rmsprop in Theano
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - stepsize * g))
    return updates


def flip_filter(w):
    if w.ndim == 4:
        t = w.copy()
        s = t.shape
        for i in range(0, s[0]):
            for j in range(0, s[1]):
                t[i][j] = np.fliplr(t[i][j])
                t[i][j] = np.flipud(t[i][j])
        return t
    else:
        return w


class ConvLayer(object):
    """Pool Layer of a convolutional network, copied from Theano tutorial """
    def __init__(self, input_tensor, filter_shape, image_shape, poolsize=(2, 2)):
        assert image_shape[1] == filter_shape[1]
        self.input = input_tensor
        fan_in = np.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) //
                   np.prod(poolsize))

        # initialize weights with random weights
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            np.asarray(np.random.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                       dtype=theano.config.floatX),
        )
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv2D_keep_shape(
            x=input_tensor,
            w=self.W,
            image_shape=image_shape,
            filter_shape=filter_shape
        )

        # downsample each feature map individually, using maxpooling
        pooled_out = theano.tensor.signal.pool.pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.nnet.relu(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        self.out_shape = (image_shape[0], filter_shape[0],
                          int(np.floor(image_shape[2]/poolsize[0])),
                          int(np.floor(image_shape[3]/poolsize[1])))

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input_tensor

