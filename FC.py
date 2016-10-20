# Based on tutorial by Alec Radford
# https://github.com/Newmu/Theano-Tutorials/blob/master/4_modern_net.py

import cgt
from cgt import nn
from cgt.distributions import categorical
from NNobj import *


class FC(NNobj):
    "Class for a multi-layer perceptron (fully connected network) object"
    def __init__(self, model="dense", im_size=[28, 28], dropout=True, devtype="cpu", grad_check=True, reg=0):
        if grad_check: cgt.set_precision("quad")
        self.model = model
        self.reg = reg
        np.random.seed(0)
        cgt.update_config(default_device=cgt.core.Device(devtype=devtype), backend="native")
        print(model)
        # MLP with 1 hidden layer
        if model == "dense1":
            self.Xsize = 2*im_size[0]*im_size[1]+im_size[0]+im_size[1]
            self.X = cgt.matrix("X", fixed_shape=(None, self.Xsize))
            self.y = cgt.vector("y", dtype='i8')
            self.p_drop_input, self.p_drop_hidden = (0.2, 0.5) if dropout else (0, 0)
            self.w_h = init_weights(self.Xsize, 256)
            self.w_o = init_weights(256, 8)
            self.pofy_drop = dense_model1(self.X, self.w_h, self.w_o, self.p_drop_input, self.p_drop_hidden)
            self.pofy_nodrop = dense_model1(self.X, self.w_h, self.w_o, 0., 0.)
            self.params = [self.w_h, self.w_o]
            self.l1 = cgt.abs(self.w_h).sum() + cgt.abs(self.w_o).sum()
            self.cost_drop = -cgt.mean(categorical.loglik(self.y, self.pofy_drop)) + self.reg*self.l1
        # MLP with 2 hidden layers
        elif model == "dense2":
            self.Xsize = 2*im_size[0]*im_size[1]+im_size[0]+im_size[1]
            self.X = cgt.matrix("X", fixed_shape=(None, self.Xsize))
            self.y = cgt.vector("y", dtype='i8')
            self.p_drop_input, self.p_drop_hidden = (0.2, 0.5) if dropout else (0, 0)
            self.w_h = init_weights(self.Xsize, 256)
            self.w_h2 = init_weights(256, 256)
            self.w_o = init_weights(256, 8)
            self.pofy_drop = dense_model2(self.X, self.w_h, self.w_h2, self.w_o, self.p_drop_input, self.p_drop_hidden)
            self.pofy_nodrop = dense_model2(self.X, self.w_h, self.w_h2, self.w_o, 0., 0.)
            self.params = [self.w_h, self.w_h2, self.w_o]
            self.l1 = cgt.abs(self.w_h).sum() + cgt.abs(self.w_h2).sum() + cgt.abs(self.w_o).sum()
            self.cost_drop = -cgt.mean(categorical.loglik(self.y, self.pofy_drop)) + self.reg*self.l1
        # MLP with 3 hidden layers
        elif model == "dense3":
            self.Xsize = 2*im_size[0]*im_size[1]+im_size[0]+im_size[1]
            self.X = cgt.matrix("X", fixed_shape=(None, self.Xsize))
            self.y = cgt.vector("y", dtype='i8')
            self.p_drop_input, self.p_drop_hidden = (0.0, [0.5, 0.5, 0.5]) if dropout else (0, [0, 0, 0])
            self.w_h = init_weights(self.Xsize, 256)
            self.w_h2 = init_weights(256, 256)
            self.w_h3 = init_weights(256, 256)
            self.w_o = init_weights(256, 8)
            self.pofy_drop = dense_model3(self.X, self.w_h, self.w_h2, self.w_h3, self.w_o, self.p_drop_input,
                                          self.p_drop_hidden)
            self.pofy_nodrop = dense_model3(self.X, self.w_h, self.w_h2, self.w_h3, self.w_o, 0., [0., 0., 0.])
            self.params = [self.w_h, self.w_h2, self.w_h3, self.w_o]
            self.l1 = cgt.abs(self.w_h).sum() + cgt.abs(self.w_h2).sum() + cgt.abs(self.w_h3).sum() + \
                      cgt.abs(self.w_o).sum()
            self.cost_drop = -cgt.mean(categorical.loglik(self.y, self.pofy_drop)) + self.reg*self.l1
        else:
            raise RuntimeError("Unknown Model")

        self.y_nodrop = cgt.argmax(self.pofy_nodrop, axis=1)
        self.cost_nodrop = -cgt.mean(categorical.loglik(self.y, self.pofy_nodrop))
        self.err_nodrop = cgt.cast(cgt.not_equal(self.y_nodrop, self.y), cgt.floatX).mean()
        self.computeloss = cgt.function(inputs=[self.X, self.y], outputs=[self.err_nodrop,self.cost_nodrop])
        self.y_out = cgt.function(inputs=[self.X], outputs=[self.y_nodrop])
        self.updates = rmsprop_updates(self.cost_drop, self.params)
        self.train = cgt.function(inputs=[self.X, self.y], outputs=[], updates=self.updates)

    def run_training(self, input, stepsize=0.01, epochs=10, output='None', batch_size=128, grad_check=True,
                     profile=False, step_decrease_rate=0.5, step_decrease_time=1000):
        # run NN training from input matlab data file, and save test data prediction in output file

        # load data from Matlab file, including
        # im_data: flattened images
        # state_data: concatenated one-hot vectors for each state variable
        # label_data: one-hot vector for action (state difference)
        if grad_check: cgt.set_precision("quad")
        matlab_data = sio.loadmat(input)
        im_data = matlab_data["im_data"]
        im_data = (im_data - 1)/255  # obstacles = 1, free zone = 0
        state_data = matlab_data["state_data"]
        value_data = matlab_data["value_data"]
        label_data = matlab_data["label_data"]
        Xdata = (np.concatenate((np.concatenate((im_data,value_data),axis=1), state_data), axis=1)).astype(cgt.floatX)
        ydata = label_data

        training_samples = int(6/7.0*Xdata.shape[0])
        Xtrain = Xdata[0:training_samples]
        ytrain = ydata[0:training_samples]

        Xtest = Xdata[training_samples:]
        ytest = ydata[training_samples:]

        sortinds = np.random.permutation(training_samples)
        Xtrain = Xtrain[sortinds]
        ytrain = ytrain[sortinds]

        self.updates = rmsprop_updates(self.cost_drop, self.params, stepsize=stepsize)
        self.train = cgt.function(inputs=[self.X, self.y], outputs=[], updates=self.updates)

        from cgt.tests import gradcheck_model
        if grad_check:
            cost_nodrop = cgt.core.clone(self.cost_nodrop, {self.X: Xtrain[:1], self.y: ytrain[:1]})
            print "doing gradient check..."
            print "------------------------------------"
            gradcheck_model(cost_nodrop, self.params[0:1])
            print "success!"
            return

        if profile: cgt.profiler.start()

        print fmt_row(10, ["Epoch","Train NLL","Train Err","Test NLL","Test Err","Epoch Time"])
        for i_epoch in xrange(int(epochs)):
            tstart = time.time()
            for start in xrange(0, Xtrain.shape[0], batch_size):
                end = start+batch_size
                self.train(Xtrain[start:end], ytrain[start:end])
            elapsed = time.time() - tstart
            trainerr, trainloss = self.computeloss(Xtrain[:len(Xtest)], ytrain[:len(Xtest)])
            testerr, testloss = self.computeloss(Xtest, ytest)
            print fmt_row(10, [i_epoch, trainloss, trainerr, testloss, testerr, elapsed])
            if (i_epoch > 0) & (i_epoch % step_decrease_time == 0):
                stepsize = step_decrease_rate * stepsize
                self.updates = rmsprop_updates(self.cost_drop, self.params, stepsize=stepsize)
                self.train = cgt.function(inputs=[self.X, self.y], outputs=[], updates=self.updates)
                print stepsize
        if profile: cgt.execution.profiler.print_stats()

        # save Matlab data
        if output != 'None':
            sio.savemat(file_name=output, mdict={'in': Xtest, 'out': self.y_out(Xtest)})

    def predict(self, input):
        # NN output for a single input, read from file
        matlab_data = sio.loadmat(input)
        im_data = matlab_data["im_data"]
        im_data = (im_data - 1)/255  # obstacles = 1, free zone = 0
        state_data = matlab_data["state_data"]
        value_data = matlab_data["value_data"]
        x_test = (np.concatenate((np.concatenate((im_data, value_data), axis=1), state_data), axis=1)).astype(cgt.floatX)
        out = self.y_out(x_test)
        return out[0][0]


def init_weights(*shape):
    return cgt.shared(np.random.randn(*shape) * 0.01, fixed_shape_mask='all')


def rmsprop_updates(cost, params, stepsize=0.001, rho=0.9, epsilon=1e-6):
    grads = cgt.grad(cost, params)
    updates = []
    for p, g in zip(params, grads):
        acc = cgt.shared(p.op.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * cgt.square(g)
        gradient_scaling = cgt.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - stepsize * g))
    return updates


def adagrad_updates(cost, params, stepsize=0.001, rho=0.9, epsilon=1e-6):
    grads = cgt.grad(cost, params)
    updates = []
    for param, grad in zip(params, grads):
        value = param.op.get_value()
        accu = cgt.shared(np.zeros(value.shape, dtype=value.dtype))
        delta_accu = cgt.shared(np.zeros(value.shape, dtype=value.dtype))

        accu_new = rho * accu + (1 - rho) * grad ** 2
        updates.append((accu, accu_new))

        update = (grad * cgt.sqrt(delta_accu + epsilon) / cgt.sqrt(accu_new + epsilon))
        updates.append((param, param - stepsize * update))

        delta_accu_new = rho * delta_accu + (1 - rho) * update ** 2
        updates.append((delta_accu, delta_accu_new))
    return updates


def dense_model1(X, w_h, w_o, p_drop_input, p_drop_hidden):
    X = nn.dropout(X, p_drop_input)
    h = nn.rectify(cgt.dot(X, w_h))
    h = nn.dropout(h, p_drop_hidden)
    py_x = nn.softmax(cgt.dot(h, w_o))
    return py_x


def dense_model2(X, w_h, w_h2, w_o, p_drop_input, p_drop_hidden):
    X = nn.dropout(X, p_drop_input)
    h = nn.rectify(cgt.dot(X, w_h))

    h = nn.dropout(h, p_drop_hidden)
    h2 = nn.rectify(cgt.dot(h, w_h2))

    h2 = nn.dropout(h2, p_drop_hidden)
    py_x = nn.softmax(cgt.dot(h2, w_o))
    return py_x


def dense_model3(X, w_h, w_h2, w_h3, w_o, p_drop_input, p_drop_hidden):
    X = nn.dropout(X, p_drop_input)
    h = nn.rectify(cgt.dot(X, w_h))

    h = nn.dropout(h, p_drop_hidden[0])
    h2 = nn.rectify(cgt.dot(h, w_h2))

    h2 = nn.dropout(h2, p_drop_hidden[1])
    h3 = nn.rectify(cgt.dot(h2, w_h3))

    h3 = nn.dropout(h3, p_drop_hidden[2])
    py_x = nn.softmax(cgt.dot(h3, w_o))
    return py_x


