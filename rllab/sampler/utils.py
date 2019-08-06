import numpy as np
from rllab.misc import tensor_utils
import time


def rollout(env, agent, max_path_length=np.inf, animated=False, speedup=1,
            always_return_paths=False, deterministic=False):
    observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []
    o = env.wrapped_env.reset(create_log=False)
    agent.reset()
    path_length = 0
    if animated:
        env.render()
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        if deterministic: a = agent_info["mean"]
        next_o, r, d, env_info = env.step(a)
        observations.append(env.observation_space.flatten(o))
        rewards.append(r)
        actions.append(env.action_space.flatten(a))
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
        if animated:
            print('rendering env')
            env.render()
            timestep = 0.05
            time.sleep(timestep / speedup)
    if animated and not always_return_paths:
        return

    # Reset environment at the end s.t. teststand doesn't collide.
    # print('We should call reset at the end of episode from rllab/sampler/utils.py, s.t. teststand doesnt collide')
    env.reset()
    
    return dict(
        observations=tensor_utils.stack_tensor_list(observations),
        actions=tensor_utils.stack_tensor_list(actions),
        rewards=tensor_utils.stack_tensor_list(rewards),
        agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
        env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
    )
