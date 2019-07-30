import os
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
os.environ["THEANO_FLAGS"] = "device=cpu"
#from rllab.sandbox.vime.envs.cartpole_swingup_env_x import CartpoleSwingupEnvX
#from rllab.envs.box2d.mountain_car_env import MountainCarEnv
#from rllab.envs.mujoco.gather.swimmer_gather_env import SwimmerGatherEnv
#import sys
#sys.path.append("..") # Adds higher directory to python modules path.

from rllab.sandbox.vime.envs.stand_env_vime import StandEnvVime
#from envs.stand_env_vime import StandEnvVime


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

# Params for testing 
#small_neg_rew=False
partial_obs = None#'err_only' #'height_only', None
sim = "sim_physics" # "sim", "real"
#t_lookahead=0
#t_past=0
#init_w_lqt=False
dead_band=550.
max_action = 900.
vis = False
verbose = False

for step_size in [0.01, 0.005]:

    if partial_obs:
        policy_net_size = (2,2)
    else:
        policy_net_size = (64, 32)
    print('net size', policy_net_size)

    mdp_classes = [StandEnvVime]#[SwimmerGatherEnv]
    mdps = [NormalizedEnv(env=mdp_class(
        timeout=timeout,
        sim=sim,
        vis=vis,
        verbose=verbose, 
        dead_band=dead_band,
        max_action=max_action
    )) for mdp_class in mdp_classes]
    param_cart_product = itertools.product(
        mdps, seeds
    )

    for mdp, seed in param_cart_product:
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
        
        batch_size = 5000#5000
        n_itr = 500
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
            log_dir="data/logs_trpo_sim_physics_st_sz_%.3f_sd_%d_db_%3.0f_max_%3.0f"%(step_size, seed, dead_band,max_action),
            plot=False,
            script="rllab/sandbox/vime/experiments/run_experiment_lite.py"
        )

## VIME
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.sandbox.vime.algos.trpo_expl import TRPO as Trpo_vime

stub(globals())

# Param ranges
etas = [0.0001]
# Init env
timeout = 0.02

# Params for testing 
#small_neg_rew=False
partial_obs = None#'err_only' #'height_only', None
sim = "sim_physics" # "sim", "real"
#t_lookahead=0
#t_past=0
#init_w_lqt=False
#elim_dead_bands=False


#reward_fn = lambda state, action, next_state: reward(state, action, next_state)
mdp_classes = [StandEnvVime]#[SwimmerGatherEnv]
mdps = [NormalizedEnv(env=mdp_class(
    timeout=timeout, 
    sim=sim,
    vis=vis,
    verbose=verbose, 
    dead_band=dead_band,
    max_action=max_action
    )) for mdp_class in mdp_classes]

seeds = range(1)#range(5)
param_cart_product = itertools.product(
    mdps, etas, seeds
)

for step_size in [0.01, 0.005]:
    for mdp, eta, seed in param_cart_product:
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
        batch_size = 5000#5000
        n_itr = 2000#150
        algo = Trpo_vime(
            env=mdp,
            policy=policy,
            baseline=baseline,
            batch_size=batch_size,
            whole_paths=True, 
            max_path_length=500,
            n_itr=n_itr,
            step_size=0.01,
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
            log_dir="data/logs_vime_sim_physics_st_sz_%.3f_sd_%d_db_%3.0f_max_%3.0f"%(step_size, seed, dead_band,max_action),
            script="rllab/sandbox/vime/experiments/run_experiment_lite.py",
        )

