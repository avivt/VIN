# interface for NN object

import numpy as np
import pickle
import scipy.io as sio
import time


class NNobj:
    "Class for a multi-layer perceptron object"
    def __init__(self):
        raise RuntimeError("Not implemented")

    def save_weights(self, outfile="weight_dump.pk"):
        pickle.dump([n.op.get_value() for n in self.params], open(outfile, 'w'))

    def load_weights(self, infile="weight_dump.pk"):
        dump = pickle.load(open(infile, 'r'))
        [n.op.set_value(p) for n, p in zip(self.params, dump)]


# helper methods to print nice table (taken from CGT code)
def fmt_item(x, l):
    if isinstance(x, np.ndarray):
        assert x.ndim==0
        x = x.item()
    if isinstance(x, float): rep = "%g"%x
    else: rep = str(x)
    return " "*(l - len(rep)) + rep


def fmt_row(width, row, header=False):
    out = " | ".join(fmt_item(x, width) for x in row)
    if header: out = out + "\n" + "-"*len(out)
    return out
