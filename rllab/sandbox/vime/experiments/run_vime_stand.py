import os
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
os.environ["THEANO_FLAGS"] = "device=cpu"
#from rllab.envs.mujoco.gather.swimmer_gather_env import SwimmerGatherEnv

#from rllab.envs.box2d.cartpole_swingup_env import Box2DEnv
#from rllab.sandbox.vime.envs.cartpole_swingup_env_x import CartpoleSwingupEnvX
from rllab.sandbox.vime.envs.stand_env_vime import StandEnvVime
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.envs.normalized_env import NormalizedEnv

from rllab.sandbox.vime.algos.trpo_expl import TRPO
from rllab.misc.instrument import stub, run_experiment_lite
import itertools

from solenoid.misc.tasks import SineTask,ChirpTask,StepTask

stub(globals())
import atexit

## VIME
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.sandbox.vime.algos.trpo_expl import TRPO as Trpo_vime

stub(globals())

"""
Train TRPO+VIME on teststand environment
"""

# Param ranges
seeds = [0]#range(1)
timeout = 0.02

# Define goal task
periods = 1.
task = SineTask(
    steps=500, periods=periods, offset=0.)
#reward_fn = goal_bias_action_penalty_2
#small_neg_rew=False
partial_obs = None#'err_only' #'height_only', None
sim = "sim_physics"#"real" # "sim", "real"
#t_lookahead=0
#t_past=0
#init_w_lqt=False
dead_band=550.
max_action = 900.#1700.#900.
vis = False
verbose = False
learn_lqt_plus_rl = False
        
batch_size = 5000
n_itr = 500#150#1500

# VIME reward weighting
eta = 0.0001

for step_size in [0.005, 0.01, 0.001]:
    for seed in seeds:
        np.random.seed(seed)
        random.seed(seed)

        log_dir="logs_vime_%s_st_sz_%.3f_sd_%d_db_%3.0f_max_%3.0f_lqt_%r"%(sim, step_size, seed, dead_band,max_action, learn_lqt_plus_rl)

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

        if partial_obs:
            policy_net_size = (2,2)
        else:
            policy_net_size = (64, 32)

        policy = GaussianMLPPolicy(
            env_spec=mdp.spec,
            hidden_sizes=policy_net_size,
        )

        baseline = LinearFeatureBaseline(
            mdp.spec,
        )

        plot = False

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
        # Run experiment lite in "pickled mode" which is necessary for parallelization
        run_experiment_lite(
            algo.train(),
            sim=sim,
            use_cloudpickle=True, # to pickle lambda reward fn
            exp_prefix="vime-stand", 
            n_parallel=1,
            snapshot_mode="all",
            seed=seed,
            plot=plot,
            mode="local",
            log_dir="data/"+log_dir,
            script="rllab/sandbox/vime/experiments/run_experiment_lite.py",
        )

