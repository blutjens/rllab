import os
import numpy as np
import random
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
os.environ["THEANO_FLAGS"] = "device=cpu"
#from rllab.sandbox.vime.envs.cartpole_swingup_env_x import CartpoleSwingupEnvX
#from rllab.envs.box2d.mountain_car_env import MountainCarEnv
#from rllab.envs.mujoco.gather.swimmer_gather_env import SwimmerGatherEnv

from rllab.sandbox.vime.envs.stand_env_vime import StandEnvVime

from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.envs.normalized_env import NormalizedEnv

from rllab.algos.trpo import TRPO
from rllab.misc.instrument import stub, run_experiment_lite
import itertools

from solenoid.misc.reward_fns import goal_bias_action_penalty_2, goal_bias
from solenoid.misc.tasks import SineTask,ChirpTask,StepTask

stub(globals())
import atexit

"""
Train TRPO on teststand environment
"""

## Define environment parameters
timeout = 0.02
periods = 1.
task = SineTask(
    steps=500, periods=periods, offset=0.)
reward_fn = goal_bias # goal_bias_action_penalty_2
#small_neg_rew=False
partial_obs = None#'err_only' #'height_only', None
sim = "sim_physics"#"real" # "sim", "real"
#t_lookahead=0
#t_past=0
#init_w_lqt=False
dead_band=550. # Deadband linear mapping
max_action = 900.#1700.#900.
vis = False
verbose = False
learn_lqt_plus_rl = False
        
## Define network parameters
seeds = [1]#range(1)
batch_size = 5000
n_itr = 500#150#1500

for step_size in [0.005]:#, 0.01, 0.001]:
    for seed in seeds:
        np.random.seed(seed)
        random.seed(seed)

        log_dir="logs_trpo_%s_st_sz_%.3f_sd_%d_db_%03.0f_max_%3.0f_lqt_%r_rew_%s"%(sim, step_size, seed, dead_band,max_action, learn_lqt_plus_rl, reward_fn.__name__)

        if partial_obs:
            policy_net_size = (2,2)
        else:
            policy_net_size = (64, 32)

        # Initialize environment
        mdp_class = StandEnvVime
        mdp = NormalizedEnv(env=mdp_class(
            timeout=timeout,
            reward_fn = reward_fn,
            sim=sim,
            vis=vis,
            verbose=verbose, 
            dead_band=dead_band,
            max_action=max_action,
            learn_lqt_plus_rl=learn_lqt_plus_rl,
            task=task,
            clear_logdir=True,
            log_dir = "runs/"+log_dir
        ))

        # Terminate env if task is closed from terminal (e.g., ctrl+c)
        # TODO this does not work, because of stub'ed()
        atexit.register(mdp.wrapped_env.terminate())

        # Initialize policy network
        policy = GaussianMLPPolicy(
            env_spec=mdp.spec,
            hidden_sizes=policy_net_size,
        )

        # Initialize baseline
        baseline = LinearFeatureBaseline(
            mdp.spec,
        )

        # Initialize TRPO algorithm
        algo = TRPO(
            env=mdp,
            policy=policy,
            baseline=baseline,
            batch_size=batch_size,
            whole_paths=True,
            max_path_length=500,#500,
            n_itr=n_itr,#10000,
            step_size=step_size,#0.01,
            subsample_factor=1.0,
            plot=False
        )

        # Popolate parallelizable workers with training tasks
        run_experiment_lite(
            algo.train(),
            sim=sim,
            use_cloudpickle=True, # to pickle lambda reward fn
            exp_prefix="trpo",
            n_parallel=1, # Run multiple rollouts in parallel (TODO: fix tensorboard iteration counter)
            snapshot_mode="all", # Store network checkpoint at every training iteration
            seed=seed,
            mode="local",
            log_dir="data/"+log_dir,
            plot=False, 
            script="rllab/sandbox/vime/experiments/run_experiment_lite.py"
        )
