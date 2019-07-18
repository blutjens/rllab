import os
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
os.environ["THEANO_FLAGS"] = "device=cpu"
#from rllab.sandbox.vime.envs.cartpole_swingup_env_x import CartpoleSwingupEnvX
#from rllab.envs.box2d.mountain_car_env import MountainCarEnv
from rllab.sandbox.vime.envs.stand_env_vime import StandEnvVime
#from rllab.envs.mujoco.gather.swimmer_gather_env import SwimmerGatherEnv

from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.envs.normalized_env import NormalizedEnv

from rllab.algos.trpo import TRPO
from rllab.misc.instrument import stub, run_experiment_lite
import itertools


from rllab.envs.envs.tasks import SineTask,ChirpTask,StepTask

stub(globals())



def reward(state, action, next_state):
    """Reward Function for Teststand Environment"""
    # Stand-in reward function
    reward = 0.0
    return reward

# Param ranges
seeds = range(1)
# Init env
timeout = 0.02
#reward_fn = lambda state, action, next_state: reward(state, action, next_state)
mdp_classes = [StandEnvVime]#[SwimmerGatherEnv]
mdps = [NormalizedEnv(env=mdp_class(timeout=timeout)) for mdp_class in mdp_classes]
param_cart_product = itertools.product(
    mdps, seeds
)

for mdp, seed in param_cart_product:

    policy = GaussianMLPPolicy(
        env_spec=mdp.spec,
        hidden_sizes=(64, 32),
    )

    baseline = LinearFeatureBaseline(
        mdp.spec,
    )
    
    batch_size = 5000
    algo = TRPO(
        env=mdp,
        policy=policy,
        baseline=baseline,
        batch_size=batch_size,
        whole_paths=True,
        max_path_length=500 ,#500,
        n_itr=10000,#10000,
        step_size=0.01,
        subsample_factor=1.0,
        plot=True
    )

    run_experiment_lite(
        algo.train(),
        use_cloudpickle=True, # to pickle lambda reward fn
        exp_prefix="trpo",
        n_parallel=1,
        snapshot_mode="all",
        seed=seed,
        mode="local",
        log_dir="data/logs_trpo_stand",
        plot=True,
        script="rllab/sandbox/vime/experiments/run_experiment_lite.py"
    )
