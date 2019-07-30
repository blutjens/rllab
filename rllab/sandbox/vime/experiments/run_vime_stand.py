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

#from rllab.envs.envs.tasks import SineTask,ChirpTask,StepTask
from solenoid.misc.tasks import SineTask,ChirpTask,StepTask

stub(globals())
import atexit



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
    vis=True,
    verbose=False
    )) for mdp_class in mdp_classes]

seeds = range(5)
param_cart_product = itertools.product(
    mdps, etas, seeds
)

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
    batch_size = 5000
    n_itr = 150
    algo = TRPO(
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
        log_dir="data/logs_vime_sim_physics_%d"%(seed),
        script="rllab/sandbox/vime/experiments/run_experiment_lite.py",
    )

