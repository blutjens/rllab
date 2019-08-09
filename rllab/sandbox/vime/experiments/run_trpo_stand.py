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

from solenoid.misc.tasks import SineTask,ChirpTask,StepTask
#from rllab.envs.envs.tasks import SineTask,ChirpTask,StepTask

stub(globals())
import atexit

# Param ranges
seeds = range(1)
# Init env
timeout = 0.02

# Define goal task
periods = 1.
task = SineTask(
    steps=500, periods=periods, offset=0.)

# Params for testing 
#small_neg_rew=False
partial_obs = None#'err_only' #'height_only', None
sim = "sim_physics"#"real" # "sim", "real"
#t_lookahead=0
#t_past=0
#init_w_lqt=False
dead_band=0.#550.
max_action = 900. #1700.
vis = False
verbose = False
learn_lqt_plus_rl = False

for step_size in [0.005, 0.01, 0.001]:
    for seed in seeds:
        np.random.seed(seed)
        random.seed(seed)

        log_dir="logs_trpo_%s_st_sz_%.3f_sd_%d_db_%03.0f_max_%3.0f_lqt_%r"%(sim, step_size, seed, dead_band,max_action, learn_lqt_plus_rl)
        if partial_obs:
            policy_net_size = (2,2)
        else:
            policy_net_size = (64, 32)
        print('net size', policy_net_size)

        mdp_class = StandEnvVime#[SwimmerGatherEnv]
        mdp = NormalizedEnv(env=mdp_class(
            timeout=timeout,
            sim=sim,
            vis=vis,
            verbose=verbose, 
            dead_band=dead_band,
            max_action=max_action,
            learn_lqt_plus_rl=learn_lqt_plus_rl,
            clear_logdir=True,
            log_dir = "runs/"+log_dir
        ))

        # Terminate env if task is closed from terminal (e.g., ctrl+c)
        # TODO this does not work!!
        atexit.register(mdp.wrapped_env.terminate())

        policy = GaussianMLPPolicy(
            env_spec=mdp.spec,
            hidden_sizes=policy_net_size,
        )

        baseline = LinearFeatureBaseline(
            mdp.spec,
        )
        
        batch_size = 5000
        n_itr = 300#1500
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

        run_experiment_lite(
            algo.train(),
            sim=sim,
            use_cloudpickle=True, # to pickle lambda reward fn
            exp_prefix="trpo",
            n_parallel=1,
            snapshot_mode="all",
            seed=seed,
            mode="local",
            log_dir="data/"+log_dir,
            plot=False,
            script="rllab/sandbox/vime/experiments/run_experiment_lite.py"
        )
        import sys
        sys.exit()
## VIME
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.sandbox.vime.algos.trpo_expl import TRPO as Trpo_vime

stub(globals())

# Param ranges
eta = 0.0001

# Params for testing 
#small_neg_rew=False
partial_obs = None#'err_only' #'height_only', None
#t_lookahead=0
#t_past=0
#init_w_lqt=False
#elim_dead_bands=False

seeds = range(1)#range(5)

for step_size in [0.005, 0.01, 0.001]:
    for seed in seeds:
        np.random.seed(seed)
        random.seed(seed)

        log_dir="logs_vime_%s_st_sz_%.3f_sd_%d_db_%3.0f_max_%3.0f_lqt_%r"%(sim, step_size, seed, dead_band,max_action, learn_lqt_plus_rl)

        #reward_fn = lambda state, action, next_state: reward(state, action, next_state)
        mdp_class = StandEnvVime#[SwimmerGatherEnv]
        mdp = NormalizedEnv(env=mdp_class(
            timeout=timeout, 
            sim=sim,
            vis=vis,
            verbose=verbose, 
            dead_band=dead_band,
            max_action=max_action,
            clear_logdir=True,
            learn_lqt_plus_rl=learn_lqt_plus_rl,
            log_dir="runs/"+log_dir,
            task=task
            ))

        # Terminate env if task is closed from terminal (e.g., ctrl+c)
        atexit.register(mdp.wrapped_env.terminate())

        policy = GaussianMLPPolicy(
            env_spec=mdp.spec,
            hidden_sizes=(64, 32),
        )

        baseline = LinearFeatureBaseline(
            mdp.spec,
        )

        plot = False
        #batch_size = 5000#5000
        #n_itr = 2000#150
        algo = Trpo_vime(
            env=mdp,
            policy=policy,
            baseline=baseline,
            batch_size=batch_size,
            whole_paths=True, 
            max_path_length=500,
            n_itr=n_itr,
            step_size=step_size,
            eta=eta,
            snn_n_samples=10,
            subsample_factor=1.0,
            use_replay_pool=True,
            use_kl_ratio=True,
            use_kl_ratio_q=True,
            n_itr_update=1,
            kl_batch_size=1,
            normalize_reward=False,
            replay_pool_size=1000000,
            n_updates_per_sample=5000,
            second_order_update=True,
            unn_n_hidden=[32],
            unn_layers_type=[1, 1],
            plot=plot,
            unn_learning_rate=0.0001,
            animated=False
        )
        # Run experiment lite is "pickled mode" which is necessary for parallelization
        run_experiment_lite(
            algo.train(),
            sim=sim,
            use_cloudpickle=True, # to pickle lambda reward fn
            exp_prefix="vime-stand", #trpo-expl
            n_parallel=1,
            snapshot_mode="all",
            seed=seed,
            plot=plot,
            mode="local",
            log_dir="data/"+log_dir,
            script="rllab/sandbox/vime/experiments/run_experiment_lite.py",
        )

