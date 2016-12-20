from NNobj import *
from vin import vin
from vin_untied import vin_untied
from FCN import fcn
from CNN import cnn


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    parser.add_argument("--output", default="None")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--dropout", action="store_true")
    parser.add_argument("--stepsize", type=float, default=.0002)
    parser.add_argument("--model",
                        choices=["dense1", "dense2", "dense3", "conv", "valIterMultiBatch", "valIterBatch",
                                 "valIterMars", "valIterMarsSingle", "valIterBatchUntied", "fcn", "cnn"],
                        default="dense")
    parser.add_argument("--unittest", action="store_true")
    parser.add_argument("--grad_check", action="store_true")
    parser.add_argument("--devtype", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--warmstart", default="None")
    parser.add_argument("--reg", type=float, default=.0)
    parser.add_argument("--imsize", type=int, default=28)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--batchsize", type=int, default=128)
    parser.add_argument("--statebatchsize", type=int, default=1)
    parser.add_argument("--stepdecreaserate", type=float, default=1.0)
    parser.add_argument("--stepdecreasetime", type=int, default=10000)
    parser.add_argument("--data_fraction", type=float, default=1.0)
    args = parser.parse_args()

    if args.model == "fcn":
        # FCN network
        my_nn = fcn(model=args.model, im_size=[args.imsize, args.imsize], dropout=args.dropout,
                    devtype=args.devtype, grad_check=args.grad_check, reg=args.reg,
                    batchsize=args.batchsize, statebatchsize=args.statebatchsize)
    elif args.model == "cnn":
        # FCN network
        my_nn = cnn(model=args.model, im_size=[args.imsize, args.imsize], dropout=args.dropout,
                    devtype=args.devtype, grad_check=args.grad_check, reg=args.reg,
                    batchsize=args.batchsize)
    elif args.model == "valIterBatch":
        # VI network
        my_nn = vin(model=args.model, im_size=[args.imsize, args.imsize], dropout=args.dropout,
                    devtype=args.devtype, grad_check=args.grad_check, reg=args.reg, k=args.k,
                    batchsize=args.batchsize, statebatchsize=args.statebatchsize)
    elif args.model == "valIterBatchUntied":
        # VI network with untied weights
        my_nn = vin_untied(model=args.model, im_size=[args.imsize, args.imsize], dropout=args.dropout,
                           devtype=args.devtype, grad_check=args.grad_check, reg=args.reg, k=args.k,
                           batchsize=args.batchsize, statebatchsize=args.statebatchsize)
    else:
        # FC network
        my_nn = NNobj(model=args.model, im_size=[args.imsize, args.imsize], dropout=args.dropout,
                      devtype=args.devtype, grad_check=args.grad_check, reg=args.reg)
    if args.warmstart != "None":
        print('warmstarting...')
        my_nn.load_weights(args.warmstart)
    my_nn.run_training(input=str(args.input), stepsize=args.stepsize, epochs=args.epochs,
                       grad_check=args.grad_check, batch_size=args.batchsize, data_fraction=args.data_fraction)
    my_nn.save_weights(outfile=str(args.output))

if __name__ == "__main__":
    main()
