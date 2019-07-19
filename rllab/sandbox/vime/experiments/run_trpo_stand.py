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
small_neg_rew=True
partial_obs = 'err_only' #'height_only', None
t_lookahead=0
t_past=0
init_w_lqt=False
elim_dead_bands=False

if partial_obs:
    policy_net_size = (2,2)
else:
    policy_net_size = (64, 32)
print('net size', policy_net_size)

mdp_classes = [StandEnvVime]#[SwimmerGatherEnv]
mdps = [NormalizedEnv(env=mdp_class(
    timeout=timeout,
    small_neg_rew=small_neg_rew,
    partial_obs=partial_obs,
    t_lookahead=t_lookahead,
    t_past=t_past,
    init_w_lqt=init_w_lqt,
    elim_dead_bands=elim_dead_bands)) for mdp_class in mdp_classes]
param_cart_product = itertools.product(
    mdps, seeds
)

for mdp, seed in param_cart_product:
    # Terminate env if task is closed from terminal (e.g., ctrl+c)
    #print('mdp', mdp.wrapped_env.terminate())
    #mdp.terminate()
    #mdp.wrapped_env.terminate()
    #import sys
    #sys.exit()
    atexit.register(mdp.wrapped_env.terminate())

    policy = GaussianMLPPolicy(
        env_spec=mdp.spec,
        hidden_sizes=policy_net_size,
    )

    baseline = LinearFeatureBaseline(
        mdp.spec,
    )
    
    batch_size = 2000
    n_itr = 100
    plot = False
    algo = TRPO(
        env=mdp,
        policy=policy,
        baseline=baseline,
        batch_size=batch_size,
        whole_paths=True,
        max_path_length=500 ,#500,
        n_itr=n_itr,#10000,
        step_size=0.01,
        subsample_factor=1.0,
        plot=plot
    )

    run_experiment_lite(
        algo.train(),
        use_cloudpickle=True, # to pickle lambda reward fn
        exp_prefix="trpo",
        n_parallel=1,
        snapshot_mode="all",
        seed=seed,
        mode="local",
        log_dir="data/logs_trpo_stand_1",
        plot=False,
        script="rllab/sandbox/vime/experiments/run_experiment_lite.py"
    )
