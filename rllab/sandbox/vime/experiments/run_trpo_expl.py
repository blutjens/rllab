import os
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
os.environ["THEANO_FLAGS"] = "device=cpu"
#from rllab.envs.mujoco.gather.swimmer_gather_env import SwimmerGatherEnv

#from rllab.envs.box2d.cartpole_swingup_env import Box2DEnv
from rllab.sandbox.vime.envs.cartpole_swingup_env_x import CartpoleSwingupEnvX
from rllab.sandbox.vime.envs.mountain_car_env_x import MountainCarEnvX
from rllab.envs.box2d.mountain_car_env import MountainCarEnv
from rllab.envs.box2d.double_pendulum_env import DoublePendulumEnv
from rllab.sandbox.vime.envs.stand_env_vime import StandEnvVime
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.envs.normalized_env import NormalizedEnv

#from rllab.sandbox.vime.algos.trpo_expl import TRPO

from rllab.sandbox.vime.algos.trpo_expl import TRPO
from rllab.misc.instrument import stub, run_experiment_lite
import itertools

stub(globals())

# Param ranges
seeds = range(2)
etas = [0.0001]
# SwimmerGather hierarchical task
mdp_classes = [StandEnvVime]#DoublePendulumEnv]#CartpoleSwingupEnvX]#[SwimmerGatherEnv]
mdps = [NormalizedEnv(env=mdp_class())
        for mdp_class in mdp_classes]
print('number of mdps', len(mdps))

param_cart_product = itertools.product(
    mdps, etas, seeds
)

for mdp, eta, seed in param_cart_product:

    policy = GaussianMLPPolicy(
        env_spec=mdp.spec,
        hidden_sizes=(64, 32),
    )

    baseline = LinearFeatureBaseline(
        mdp.spec,
    )

    plot = True
    batch_size = 50000
    algo = TRPO(
        env=mdp,
        policy=policy,
        baseline=baseline,
        batch_size=batch_size,
        whole_paths=True,
        max_path_length=500,
        n_itr=10000,
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
        unn_learning_rate=0.0001
    )
    print('skrt')
    run_experiment_lite(
        algo.train(),
        exp_prefix="trpo-expl",
        n_parallel=1,
        snapshot_mode="all",
        seed=seed,
        plot=plot,
        mode="local",
        log_dir="data/logs_tst",
        script="rllab/sandbox/vime/experiments/run_experiment_lite.py",
    )
