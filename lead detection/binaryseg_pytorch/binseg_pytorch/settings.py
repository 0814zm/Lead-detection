from copy import deepcopy
import argparse

# --- settings common to train and eval ---
defaults = argparse.Namespace()
defaults.MODEL_NAME = 'deeplab_v3_plus_mobilev2_se_sa1_lau'
defaults.LOAD_STATE = ''
defaults.ADD_MODULE = 'S'
defaults.VISDOM_PORT = 8097

# --- common settings ---
defaults_common = deepcopy(defaults)

# --- train settings ---
defaults_train = deepcopy(defaults)
defaults_train.BATCH_SIZE = 1
defaults_train.INPUT_WIDTH = 512
defaults_train.INPUT_HEIGHT = 512
defaults_train.LEARNING_RATE = 0.01
defaults_train.MOMENTUM = 0.99
defaults_train.NUM_EPOCHS = 30
# Note that rgb and gt images has the same filenames in different folders and in bmp extension
defaults_train.GT_DIR = r'D:\6_code\binary_seg\binaryseg_pytoch\binseg_pytoch-master\data\hh_dot_hv\masks_s'
defaults_train.IMG_DIR = r'D:\6_code\binary_seg\binaryseg_pytoch\binseg_pytoch-master\data\hh_dot_hv\imgs_s'
defaults_train.NUM_WORKERS = 0
defaults_train.WEIGHT_DECAY = 0.00005
defaults_train.LOG_STEP = 1
# --- crossval settings ---
defaults_crossval = deepcopy(defaults_train)
defaults_crossval.NUM_EPOCHS = 20
defaults_crossval.BATCH_SIZE = list([1, 8])
defaults_crossval.LEARNING_RATE = list([0.0001, 0.001, 0.01])
defaults_crossval.MOMENTUM = list([0.9, 0.99])
defaults_crossval.OPTIMIZER = list(['sgd', 'adam'])

#  --- eval settings ---
defaults_eval = deepcopy(defaults)
defaults_eval.IMGDIR = 'data/hh_dot_hv/test_imgs'
defaults_eval.RESDIR = 'data/hh_dot_hv/results/big_paper/Deeplab_v3_plus_mobv2_segaspp4812_sa1_lau_new_30'
def get_arguments(argv):

    defaults = defaults_common
    parser = argparse.ArgumentParser(description="")
    subparsers = parser.add_subparsers(dest='mode')
    subparsers.required = True
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument("--model", type=str, default=defaults.MODEL_NAME,
                        help="Name of the model")
    parser.add_argument("--state", type=str, required=False, default=defaults.LOAD_STATE,
                        help="Number of state to load")
    parser.add_argument("--visdom-port", type=int, required=False, default=defaults.VISDOM_PORT,
                        help="Port to use in Visdom Visualization")
    parser.add_argument("--add_module", type=str, required=False, default=defaults.ADD_MODULE,
                        help="name of the new module")

    parser_eval = subparsers.add_parser('eval')
    defaults = defaults_eval
    parser_eval.add_argument('--imgdir', type=str, default=defaults.IMGDIR)
    parser_eval.add_argument('--savedir', type=str, default=defaults.RESDIR)

    defaults = defaults_train
    parser_train = subparsers.add_parser('train')

    parser_train.add_argument("--num-epochs", type=int, default= defaults.NUM_EPOCHS,
                              help="Number of Epochs to train")
    parser_train.add_argument("--batch-size", type=int, default=defaults.BATCH_SIZE,
                              help="Number of images sent to the network in one step.")
    parser_train.add_argument("--lr", type=float, default=defaults.LEARNING_RATE,
                              help="Base learning rate for training with polynomial decay.")
    parser_train.add_argument("--momentum", type=float, default=defaults.MOMENTUM,
                              help="Momentum component of the optimiser.")
    parser_train.add_argument('--imgdir', required=False, default=defaults.IMG_DIR,
                              help="Path to the RGB images for training")
    parser_train.add_argument('--gtdir', required=False, default=defaults.GT_DIR,
                              help="Path to the GT images for training")
    parser_train.add_argument('--num-workers', type=int, default=defaults.NUM_WORKERS)
    parser_train.add_argument('--weight-decay', type=float, default=defaults.WEIGHT_DECAY)
    parser_train.add_argument('--log-step', type=int, default=defaults.LOG_STEP)
    parser_train.add_argument('--input-width', type=int, default=defaults.INPUT_WIDTH)
    parser_train.add_argument('--input-height', type=int, default=defaults.INPUT_HEIGHT)

    parser_crossval = subparsers.add_parser('crossval')
    defaults = defaults_crossval
    defaults_crossval.BATCH_SIZE = list([1, 8])
    defaults_crossval.LEARNING_RATE = list([0.0001, 0.001, 0.01])
    defaults_crossval.MOMENTUM = list([0.9, 0.99])
    defaults_crossval.OPTIMIZER = list(['sgd', 'adam'])

    parser_crossval.add_argument("--num-epochs", type=int, default=defaults.NUM_EPOCHS,
                                 help="Number of Epochs to train")
    parser_crossval.add_argument("--batch-size", nargs='+', type=int, default=defaults.BATCH_SIZE,
                                 help="Number of images sent to the network in one step.")
    parser_crossval.add_argument("--lr", nargs='+', type=float, default=defaults.LEARNING_RATE,
                                 help="Base learning rate for training with polynomial decay.")
    parser_crossval.add_argument("--momentum", nargs='+', type=float, default=defaults.MOMENTUM,
                                 help="Momentum component of the optimiser.")
    parser_crossval.add_argument("--optimizer", nargs='+', default=defaults.OPTIMIZER,
                                 help="Type of optimiser.")
    parser_crossval.add_argument('--imgdir', required=False, default=defaults.IMG_DIR,
                                 help="Path to the RGB images for training")
    parser_crossval.add_argument('--gtdir', required=False, default=defaults.GT_DIR,
                                 help="Path to the GT images for training")
    parser_crossval.add_argument('--num-workers', type=int, default=defaults.NUM_WORKERS)
    parser_crossval.add_argument('--weight-decay', type=float, default=defaults.WEIGHT_DECAY)
    parser_crossval.add_argument('--log-step', type=int, default=defaults.LOG_STEP)
    parser_crossval.add_argument('--settings-id', type=int, required=True)

    args = parser.parse_args(argv)

    return args
