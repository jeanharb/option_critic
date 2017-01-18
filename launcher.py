import os, sys
import argparse
import numpy as np
import theano
from ale_python_interface import ALEInterface
from train_agent import Q_Learning

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

def process_args(args, defaults, description):
    """
    Handle the command line.

    args     - list of command line arguments (not including executable name)
    defaults - a name space with variables corresponding to each of
               the required default command line values.
    description - a string to display at the top of the help message.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-r', '--rom', dest="rom", default=defaults.ROM,
                        help='ROM to run (default: %(default)s)')
    parser.add_argument('-e', '--epochs', dest="epochs", type=int,
                        default=defaults.EPOCHS,
                        help='Number of training epochs (default: %(default)s)')
    parser.add_argument('-s', '--steps-per-epoch', dest="steps_per_epoch",
                        type=int, default=defaults.STEPS_PER_EPOCH,
                        help='Number of steps per epoch (default: %(default)s)')
    parser.add_argument('-t', '--test-length', dest="steps_per_test",
                        type=int, default=defaults.STEPS_PER_TEST,
                        help='Number of steps per test (default: %(default)s)')
    parser.add_argument('--optimal-eps', dest='optimal_eps',
                        type=float, default=defaults.OPTIMAL_EPS,
                        help='Epsilon when playing optimally (default: %(default)s)')
    parser.add_argument('--display-screen', dest="display_screen",
                        action='store_true', default=False,
                        help='Show the game screen.')
    parser.add_argument('--testing', dest="testing",
                        action='store_true', default=False,
                        help='Signals running test.')
    parser.add_argument('--experiment-prefix', dest="experiment_prefix",
                        default=None,
                        help='Experiment name prefix '
                        '(default is the name of the game)')
    parser.add_argument('--frame-skip', dest="frame_skip",
                        default=defaults.FRAME_SKIP, type=int,
                        help='Every how many frames to process '
                        '(default: %(default)s)')

    parser.add_argument('--update-rule', dest="update_rule",
                        type=str, default=defaults.UPDATE_RULE,
                        help=('adam|adadelta|rmsprop|sgd ' +
                              '(default: %(default)s)'))
    parser.add_argument('--learning-rate', dest="learning_rate",
                        type=float, default=defaults.LEARNING_RATE,
                        help='Learning rate (default: %(default)s)')
    parser.add_argument('--rms-decay', dest="rms_decay",
                        type=float, default=defaults.RMS_DECAY,
                        help='Decay rate for rms_prop (default: %(default)s)')
    parser.add_argument('--rms-epsilon', dest="rms_epsilon",
                        type=float, default=defaults.RMS_EPSILON,
                        help='Denominator epsilson for rms_prop ' +
                        '(default: %(default)s)')
    parser.add_argument('--clip-delta', dest="clip_delta", type=float,
                        default=defaults.CLIP_DELTA,
                        help=('Max absolute value for Q-update delta value. ' +
                              '(default: %(default)s)'))
    parser.add_argument('--discount', type=float, default=defaults.DISCOUNT,
                        help='Discount rate')
    parser.add_argument('--epsilon-start', dest="epsilon_start",
                        type=float, default=defaults.EPSILON_START,
                        help=('Starting value for epsilon. ' +
                              '(default: %(default)s)'))
    parser.add_argument('--epsilon-min', dest="epsilon_min",
                        type=float, default=defaults.EPSILON_MIN,
                        help='Minimum epsilon. (default: %(default)s)')
    parser.add_argument('--epsilon-decay', dest="epsilon_decay",
                        type=float, default=defaults.EPSILON_DECAY,
                        help=('Number of steps to minimum epsilon. ' +
                              '(default: %(default)s)'))
    parser.add_argument('--phi-length', dest="phi_length",
                        type=int, default=defaults.PHI_LENGTH,
                        help=('Number of recent frames used to represent ' +
                              'state. (default: %(default)s)'))
    parser.add_argument('--max-history', dest="replay_memory_size",
                        type=int, default=defaults.REPLAY_MEMORY_SIZE,
                        help=('Maximum number of steps stored in replay ' +
                              'memory. (default: %(default)s)'))
    parser.add_argument('--batch-size', dest="batch_size",
                        type=int, default=defaults.BATCH_SIZE,
                        help='Batch size. (default: %(default)s)')
    parser.add_argument('--freeze-interval', dest="freeze_interval",
                        type=int, default=defaults.FREEZE_INTERVAL,
                        help=('Interval between target freezes. ' +
                              '(default: %(default)s)'))
    parser.add_argument('--update-frequency', dest="update_frequency",
                        type=int, default=defaults.UPDATE_FREQUENCY,
                        help=('Number of actions before each SGD update. '+
                              '(default: %(default)s)'))
    parser.add_argument('--replay-start-size', dest="replay_start_size",
                        type=int, default=defaults.REPLAY_START_SIZE,
                        help=('Number of random steps before training. ' +
                              '(default: %(default)s)'))
    parser.add_argument('--resize-method', dest="resize_method",
                        type=str, default=defaults.RESIZE_METHOD,
                        help=('crop|scale (default: %(default)s)'))
    parser.add_argument('--crop-offset', dest="offset",
                        type=str, default=defaults.OFFSET,
                        help=('crop offset.'))
    parser.add_argument('--nn-file', dest="nn_file", type=str, default=None,
                        help='Pickle file containing trained net.')
    parser.add_argument('--cap-reward', dest="do_cap_reward",
                        type=str2bool, default=defaults.CAP_REWARD,
                        help=('true|false (default: %(default)s)'))
    parser.add_argument('--death-ends-episode', dest="death_ends_episode",
                        type=str2bool, default=defaults.DEATH_ENDS_EPISODE,
                        help=('true|false (default: %(default)s)'))
    parser.add_argument('--max-start-nullops', dest="max_start_nullops",
                        type=int, default=defaults.MAX_START_NULLOPS,
                        help=('Maximum number of null-ops at the start ' +
                              'of games. (default: %(default)s)'))
    parser.add_argument('--folder-name', dest="folder_name",
                        type=str, default="",
                        help='Name of pkl files destination (within models/)')
    parser.add_argument('--termination-reg', dest="termination_reg",
                        type=float, default=defaults.TERMINATION_REG,
                        help=('Regularization to decrease termination prob.'+
                            ' (default: %(default)s)'))
    parser.add_argument('--entropy-reg', dest="entropy_reg",
                        type=float, default=defaults.ENTROPY_REG,
                        help=('Regularization to increase policy entropy.'+
                            ' (default: %(default)s)'))
    parser.add_argument('--num-options', dest="num_options",
                        type=int, default=defaults.NUM_OPTIONS,
                        help=('Number of options to create.'+
                            ' (default: %(default)s)'))
    parser.add_argument('--actor-lr', dest="actor_lr",
                        type=float, default=defaults.ACTOR_LR,
                        help=('Actor network learning rate (default: %(default)s)'))
    parser.add_argument('--double-q', dest='double_q',
                        type=str2bool, default=defaults.DOUBLE_Q,
                        help='Train using Double Q networks. (default: %(default)s)')
    parser.add_argument('--mean-frame', dest='mean_frame',
                        type=str2bool, default=defaults.MEAN_FRAME,
                        help='Use pixel-wise mean consecutive frames as images. (default: %(default)s)')
    parser.add_argument('--temp', dest='temp',
                        type=float, default=defaults.TEMP,
                        help='Action distribution softmax tempurature param. (default: %(default)s)')
    parser.add_argument('--baseline', dest='baseline',
                        type=str2bool, default=defaults.BASELINE,
                        help='use baseline in actor gradient function. (default: %(default)s)')
    parameters = parser.parse_args(args)
    print parameters
    if parameters.experiment_prefix is None:
        name = os.path.splitext(os.path.basename(parameters.rom))[0]
        parameters.experiment_prefix = name

    return parameters

def load_params(model_path):
  import pickle as pkl
  mydir = "/".join(model_path.split("/")[:-1])
  model_params = pkl.load(open(os.path.join(mydir, 'model_params.pkl'), 'rb'))
  return model_params


def launch(args, defaults, description):
    """
    Execute a complete training run.
    """

    rec_screen = ""
    if "--nn-file" in args:
      temp_params = vars(load_params(args[args.index("--nn-file")+1]))
      for p in temp_params:
        try:
          vars(defaults)[p.upper()] = temp_params[p]
        except:
          print "warning: parameter", p, "from param file doesn't exist."
      #rec_screen = args[args.index("--nn-file")+1][:-len("last_model.pkl")]+"/frames"

    parameters = process_args(args, defaults, description)

    if parameters.rom.endswith('.bin'):
        rom = parameters.rom
    else:
        rom = "%s.bin" % parameters.rom
    parameters.rom_path = os.path.join(defaults.BASE_ROM_PATH, rom)

    rng = np.random.RandomState(123456)

    folder_name = None if parameters.folder_name == "" else parameters.folder_name

    ale = ALEInterface()
    ale.setInt('random_seed', rng.randint(1000))
    ale.setBool('display_screen', parameters.display_screen)
    ale.setString('record_screen_dir', rec_screen)
    trainer = Q_Learning(model_params=parameters, ale_env=ale, folder_name=folder_name)
    trainer.train()

if __name__ == '__main__':
    pass
