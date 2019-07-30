from __future__ import print_function

import sys

sys.path.append(".")

from rllab.misc.ext import is_iterable, set_seed
from rllab.misc.instrument import concretize
from rllab import config
import rllab.misc.logger as logger
import argparse
import os.path as osp
import datetime
import dateutil.tz
import ast
import uuid
import pickle#cPickle as pickle
import base64
import joblib # to resume from ckpoint

def run_experiment(argv):


    default_log_dir = config.LOG_DIR
    now = datetime.datetime.now(dateutil.tz.tzlocal())

    # avoid name clashes when running distributed jobs
    rand_id = str(uuid.uuid4())[:5]
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S_%f_%Z')

    default_exp_name = 'experiment_%s_%s' % (timestamp, rand_id)
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_parallel', type=int, default=1,
                        help='Number of parallel workers to perform rollouts.')
    parser.add_argument(
        '--exp_name', type=str, default=default_exp_name, help='Name of the experiment.')
    parser.add_argument('--log_dir', type=str, default=default_log_dir,
                        help='Path to save the log and iteration snapshot.')
    parser.add_argument('--snapshot_mode', type=str, default='all',
                        help='Mode to save the snapshot. Can be either "all" '
                             '(all iterations will be saved), "last" (only '
                             'the last iteration will be saved), or "none" '
                             '(do not save snapshots)')
    parser.add_argument('--tabular_log_file', type=str, default='progress.csv',
                        help='Name of the tabular log file (in csv).')
    parser.add_argument('--text_log_file', type=str, default='debug.log',
                        help='Name of the text log file (in pure text).')
    parser.add_argument('--params_log_file', type=str, default='params.json',
                        help='Name of the parameter log file (in json).')
    parser.add_argument('--plot', type=ast.literal_eval, default=False,
                        help='Whether to plot the iteration results')
    parser.add_argument('--log_tabular_only', type=ast.literal_eval, default=False,
                        help='Whether to only print the tabular log information (in a horizontal format)')
    parser.add_argument('--seed', type=int,
                        help='Random seed for numpy')
    parser.add_argument('--use_cloudpickle', type=bool,
                        help='Cloudpickle can store envs that contain lambda fn')
    parser.add_argument('--args_data', type=str,
                        help='Pickled data for stub objects')
    # BRT + vime
    parser.add_argument('--resume_from', type=str, default=None,
                    help='Name of the pickle file to resume experiment from.')
    parser.add_argument('--sim', type=str, default="sim",
                        choices=['sim','sim_physics','real'], help='Name of teststand to run on')

    args = parser.parse_args(argv[1:])

    if args.sim=="sim":
        # Only necessary for simulation
        import tensorflow as tf 
        tf.enable_eager_execution()
        print('In tf eager mode to run test stand forward dynamics simulation:', tf.executing_eagerly())

    from rllab.sandbox.vime.sampler import parallel_sampler_expl as parallel_sampler
    parallel_sampler.initialize(n_parallel=args.n_parallel)

    if args.seed is not None:
        set_seed(args.seed)
        parallel_sampler.set_seed(args.seed)

    if args.plot:
        from rllab.plotter import plotter
        plotter.init_worker()

    # read from stdin
    data = pickle.loads(base64.b64decode(args.args_data))
    print('logDIR', args.log_dir)

    log_dir = args.log_dir
    # exp_dir = osp.join(log_dir, args.exp_name)
    tabular_log_file = osp.join(log_dir, args.tabular_log_file)
    text_log_file = osp.join(log_dir, args.text_log_file)
    params_log_file = osp.join(log_dir, args.params_log_file)

    logger.log_parameters_lite(params_log_file, args)
    logger.add_text_output(text_log_file)
    logger.add_tabular_output(tabular_log_file)
    prev_snapshot_dir = logger.get_snapshot_dir()
    prev_mode = logger.get_snapshot_mode()
    logger.set_snapshot_dir(log_dir)
    logger.set_snapshot_mode(args.snapshot_mode)
    logger.set_log_tabular_only(args.log_tabular_only)
    logger.push_prefix("[%s] " % args.exp_name)

    # To allow resuming from ckpoint
    if args.resume_from is not None:
        data = joblib.load(args.resume_from)
        assert 'algo' in data
        algo = data['algo']
        algo.train()
    else:
        # read from stdin
        #if args.use_cloudpickle:
        #    import cloudpickle
        #    method_call = cloudpickle.loads(base64.b64decode(args.args_data))
        #    method_call(variant_data)
        #else:
        maybe_iter = concretize(data)
        if is_iterable(maybe_iter):
            for _ in maybe_iter:
                pass

            #data = pickle.loads(base64.b64decode(args.args_data))
            #maybe_iter = concretize(data)
            #if is_iterable(maybe_iter):
            #    for _ in maybe_iter:
            #        pass

    #maybe_iter = concretize(data)
    #if is_iterable(maybe_iter):
    #    for _ in maybe_iter:
    #        pass

    logger.set_snapshot_mode(prev_mode)
    logger.set_snapshot_dir(prev_snapshot_dir)
    logger.remove_tabular_output(tabular_log_file)
    logger.remove_text_output(text_log_file)
    logger.pop_prefix()


if __name__ == "__main__":
    run_experiment(sys.argv)
