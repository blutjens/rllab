import numpy as np
import pygame
from rllab.envs.box2d.parser import find_body
import argparse
import os
os.environ["THEANO_FLAGS"] = "device=cpu"

from rllab import spaces
from rllab.core.serializable import Serializable
from rllab.envs.box2d.box2d_env import Box2DEnv
from rllab.misc import autoargs
from rllab.misc.overrides import overrides

# Standenv
import shlex
import subprocess
import time 
import copy
import warnings

from solenoid.envs.stand_bridge import StandBridge
from solenoid.envs.proxy_stand_bridge import ProxyStandBridge

from solenoid.envs.constants import state_keys
from solenoid.envs import constants
from solenoid.misc import tasks
from solenoid.misc.reward_fns import goal_bias


class StandEnvVime(Box2DEnv, Serializable):

    @autoargs.inherit(Box2DEnv.__init__)
    #@autoargs.arg("reward_fn", type=float,
    #              help="Reward function")
    def __init__(self,
                 task=None,
                 # standEnv
                 reward_fn=goal_bias, 
                 timeout=0.02,
                 vis=False,
                 beta=1., sigma=0.05, action_penalty=4, use_proxy=False,
                 small_neg_rew=False,
                 partial_obs=None,
                 t_lookahead=0,
                 t_past=0,
                 init_w_lqt=False,
                 elim_dead_bands=False,
                 *args, **kwargs):
        super(StandEnvVime, self).__init__(
            self.model_path("stand_env.xml.mako"),
            *args, **kwargs
        )
        # Whether to visualize in rllab env or use physical test stand
        self.vis = vis
        self.shutoff = False # Set this true to shutdown the test stand
        # Box2D params
        # self.obs_noise = 0 # Set to val, if additional observation noise desired
        # self.position_only = False # Set to True, if only position observation required
        self.max_weights_pos = 2
        # TODO find out what these parameters did:
        # self.goal_weights_pos = goal_weights_pos

        self.weights = find_body(self.world, "weights")
        self.initial_state = self._state # state that is copied to init Box2DEnv
        self.state = None

        # Tests to improve RL on tst stand (vime).
        self.small_neg_rew = small_neg_rew # Episode with out of bounds will receive only small neg rew
        self.partial_obs = partial_obs # Set obs to partial observation
        self.t_lookahead = t_lookahead # Set obs to [height_t, goal_height_t+1, ..., goal_height_t+t_lookahead] 
        self.t_past = t_past # Set obs to [height_t-t_past, ..., height_t, goal_height]
        self.init_w_lqt = init_w_lqt # Set actions to: u = LQT(x) + RL(x)
        self.elim_dead_bands = elim_dead_bands # Scale action space to {[-max_ma, -dead_b],[dead_b, max_ma]}

        # Define partial observation 
        if self.partial_obs=='height_only':# Set obs to [height_t, goal_height] 
            self.obs_state_ids = [state_keys.index('Height'),state_keys.index('Goal_Height')]
        elif self.partial_obs=='err_only':# Set obs to [goal_height - height_t]
            self.obs_state_ids = [state_keys.index('Err')]
        else:
            self.obs_state_ids = range(len(state_keys))

        # StandEnv
        if task is None:
            task = tasks.SineTask(
                steps=500, periods=2., offset=0.)

        self.task = task
        self.reward_fn = reward_fn
        self.beta = beta
        self.sigma = sigma
        self.action_penalty = action_penalty

        self.timestep = timeout # overwrites box2D timestep
        self._use_proxy = use_proxy

        # Keep track of timestep.
        self.t = 0

        # TODO Check if 1e6 is high enough as limit for state high and low boundaries (set in Box2D as global variable)
        #print('act and obs space:', self.action_space, self.observation_space)

        if use_proxy:
            self._bridge = ProxyStandBridge()
        else:
            # Create Bridge to physical test stand
            find_usbid_cmd = shlex.split("ls /dev/serial/by-id/")
            usb_id = "/dev/serial/by-id/" + subprocess.check_output(find_usbid_cmd).decode()
            usb_id = usb_id[0:-1]
            self._bridge = StandBridge(port=usb_id)

        self._prev_time = time.time()
        self._prev_state = None

        # LQR control for reset to a Prespecified Height
        self.K = constants.K

        # Init Box2D serializable
        Serializable.quick_init(self, locals())


    # ======== Read from test stand ==============
    def _convert_state_dict_to_np_arr(self, state_dict):
        return np.array([state_dict[key] for key in state_keys], dtype=np.float32)
        #state_list = []
        #for key in range(len(constants.state_keys)):
        #    state_list.append(state_dict[constants.state_keys[key]])
        #return np.array(state_list)

    def _get_height_rate(self, state):
        if self._prev_state is not None:
            state[state_keys.index('Height_Rate')] = state[state_keys.index('Height')] - self._prev_state[state_keys.index('Height')]
        else:
            state[state_keys.index('Height_Rate')] = 0.0

        return state

    @overrides
    def get_raw_obs(self):
        """
        Get raw/full observation from test stand, as defined in constants.state_keys
        Raw obs is used to move test stand to init position
        """
        state = copy.deepcopy(self._bridge.read_state()) # returns a dict
        #print('got raw state', state)
        state['Height_Rate'] = state['Goal_Height'] = state['Goal_Velocity'] = state['Prev_Action'] = state['Prev_Reward'] = state['Err'] = 0.
        state = self._convert_state_dict_to_np_arr(state)#np.array([state[key] for key in state_keys], dtype=np.float32)
        #print('calc raw + goal state', state)

        #print('test if height raet in state key', state_keys)
        #print('test if height raet in state key', True if 'Height_Rate' in state_keys else False)
        if 'Height_Rate' in state_keys: state = self._get_height_rate(state)  
        #print('converted into height rate', state)

        # Convert stand env dict representation into rllab list
        # self._cached_obs = state#self._convert_state_dict_to_np_arr(state)

        # Iterate timestep counter
        self.t += 1
        #print('self.t', self.t)

        # Goal
        # TODO eval if self.t or self.t+1
        self.goal_state = self.task(t=self.t + 1)
        state[state_keys.index('Goal_Height')] = self.goal_state

        # Goal velocity
        if self._prev_state is not None:
            state[state_keys.index('Goal_Velocity')] = state[state_keys.index('Goal_Height')] - self._prev_state[state_keys.index('Goal_Height')]
        else:
            state[state_keys.index('Goal_Velocity')] = 0.

        # Error
        state[state_keys.index('Err')] = state[state_keys.index('Goal_Height')] - state[state_keys.index('Height')]

        # write state to visualization
        if self.vis: self._write_state_to_vis(copy.deepcopy(state)) # Copies the initial state to the Box2D visualization 

        # TODO see if i can delete prev_state, prev_time ...
        self._prev_state = copy.deepcopy(state)
        #self._prev_time = time.time()
        return copy.deepcopy(state)

    @overrides
    def get_current_obs(self, partial_obs=None):
        """
        Filter full state observation to partial observation
        """
        state = self.get_raw_obs()
        #print('raw obs', state, state.shape)
        
        partial_obs = partial_obs if partial_obs is not None else self.partial_obs # set self.partial_obs as default if partial_obs is not specified
        if not partial_obs=='full': # Necessary s.t. move_test_stand_to_init can call step 
            state = np.array(state[self.obs_state_ids])
        return state


    # ========== Write to visualization =============
    def _write_state_to_vis(self, state=None):
        """
        Returns a fully filled initial state to be copied to Box2D environment 
        Input: state[Rod_Pressure,Base_Pressure,System_Pressure,Load_Sense_Pressure,Reservoir_Temperature,Height,Height_Rate]
        """

        s = []
        for body in self.world.bodies:
            print('body user data', body.userData)
            if body.userData["name"]=="weights" and not self.vis:
                # TODO This is very hard-coded. update this if I change visualization
                #print('forcing state onto vis: height, heigh rate', state[state_keys.index('Height')], state[state_keys.index('Height_Rate')])
                s.append(np.concatenate([
                    list((state[state_keys.index('Height')],0.)),
                    [body.angle],
                    list((state[state_keys.index('Height_Rate')],0.)),
                    [body.angularVelocity]
                ]))
            else:
                s.append(np.concatenate([
                    list(body.position),
                    [body.angle],
                    list(body.linearVelocity),
                    [body.angularVelocity]
                ]))
        return np.concatenate(s)

    # ========== Write to test stand =============
    def _move_test_stand_to_init(self, height):
        if self.shutoff: height = 0.75
        self._send_default_commands()
        #state = self._convert_state_dict_to_np_arr(copy.deepcopy(self._get_raw_obs_in_dict()))
        state = self.get_raw_obs()
        if self._use_proxy:
            state[state_keys.index('Height')] = height

        # TODO change max init steps back to 100
        max_init_steps = 100
        for i in range(max_init_steps):
            # Compute error in current height and desired setpoint
            state[state_keys.index('Height')] = state[state_keys.index('Height')] - height
            # Drive error in height to zero
            # TODO: Replace this LQR K with eliminated-dead-bands K
            #state = self._convert_state_dict_to_np_arr(state)
            action = np.dot(np.negative(self.K), np.array(state[:state_keys.index('Goal_Height')]))
            state, _, _, _ = self.step(action, verbose=False, partial_obs='full')

        if self.shutoff:
            self.terminate()
            import sys
            sys.exit()
    def _send_default_commands(self):
        """
        Send idle commands to test stand
        """
        self._bridge.send_default_commands()
        time.sleep(self.timestep)

    def _send_action(self, action, done):
        """
        Sends action to test stand and maintains constant rate
        """
        up = int(action[0])

        # Convert current(mA) to duty cycle and ensure its non-negative
        up_cycle = max(up, 0) / constants.max_ma
        down_cycle = max(-up, 0) / constants.max_ma


        if done: # Overwrite actions to zero actions when teststand is at the top or bottom height limit
            if self._prev_state[state_keys.index('Height')] < constants.height_min:
                action = action * 0.0 if up_cycle > 0 else action
                up_cycle = 0.0

            elif self._prev_state[state_keys.index('Height')] > constants.height_max:
                action = action * 0.0 if down_cycle > 0 else action
                down_cycle = 0.0


        self._current_time = time.time()
        # TODO: evaluate if moving of send_commands 8 lines higher has destroyed code 
        # TODO: write try catch for sending commands
        self._bridge.send_commands(down_cycle, up_cycle)

        # Maintain delay of 'timeout'
        if self._current_time - self._prev_time < self.timestep:
            delay = self.timestep - (self._current_time - self._prev_time)
            time.sleep(delay)
        else:
            warnings.warn("Delay exceeds set 'timeout' value")
            print("[WARNING] Delay exceeds set 'timeout' value")

        # TODO: evaluate if moving of prev_time 3 lines higher has destroyed code 
        self._prev_time = time.time()

    def close(self):
        self._bridge.close()

    # ========== Standard functions =============
    
    def step(self, action, verbose=True, partial_obs=None):
        """Perform a step of the environment"""

        #if self.is_normalized:
        #    action = action * constants.max_ma
        # TODO evaluate why done is called before state transition?
        done = self.is_current_done()

        self._send_action(action, done)

        state = self.get_current_obs(partial_obs)

        reward = self.compute_reward(action, done)

        if verbose and self.partial_obs=='height_only': print('t: %3d, u_t: %.8f x_t: %.5fm, g_t+1: %.5fm, R: %.4f'%(
            self.t, action[0], state[0],state[1], reward))
        elif verbose and self.partial_obs=='err_only': print('t: %3d, u_t: %.8f err_t: %.5fm, R: %.4f'%(
            self.t, action[0], state[0], reward))
        elif verbose: print('Act: %.8f x_t+1: %.5fm, %.5fm/s, R: %.4f'%(
            action[0], state[(state_keys.index('Height'))],
            state[(state_keys.index('Height_Rate'))], reward))

        # TODO, check if it was bad, that i have moved prev state def into get_raw_obs
        #self._prev_state = copy.deepcopy(state)

        info = {}

        return state, reward, done, info


    @overrides
    def reset(self, height=0.55, stay=False):
        """
        Reset teststand and visualization. Environment to a specific height or to the previous height

        Args:

        height (float) : specify height to which teststand must be reset
        
        stay (bool): If True, send zero actions and reset to same height as earlier position
                     If False, reset to height specified by param height
        """
        print('RESET CALLED')
        self._prev_time = time.time()
        if not stay: # Reset test 
            self._move_test_stand_to_init(height)
        # TODO: why do I need another send default commands?
        self._send_default_commands()

        # Reset timestep counter
        self.t = 1

        state = self.get_current_obs()
        print('state on reset', state)
        
        #self._invalidate_state_caches() # Resets the cache of state that was built in get_current_obs; this is probably unnecessary

        return state

    @overrides
    def is_current_done(self):
        done = False

        # TODO: set action to zero and set up_cycle, down_cycle if episode terminates
        # TODO: check if weight.position is _prev_state or _state 
        if self._prev_state[state_keys.index('Height')] < constants.height_min: # teststand too high
            done = True
            print('EPISODE DOWN')
        elif self._prev_state[state_keys.index('Height')] > constants.height_max:
            done = True
            print('EPISODE DOWN')
        return done 

    @overrides 
    def terminate(self):
        print('env terminate called')
        self.reset(height=0.75)
        self.close()

    @overrides
    def compute_reward(self, action, done):
        # Computations before stepping the world
        # TODO: check if I should store _prev_state here
        # TODO check if i should write yield instead of return
        # yield
        # Computations after stepping the world

        # TODO: evaluate why reward fn takes in prev_state, not state
        if not done:
            if self.reward_fn.__name__ == 'goal_bias_action_penalty':
                reward = self.reward_fn(
                    self._prev_state,
                    action,
                    beta=self.beta,
                    sigma=self.sigma,
                    penalty=self.action_penalty)
            else:
                reward = self.reward_fn(
                    self._prev_state,
                    action,
                    beta=self.beta,
                    sigma=self.sigma)
        else:
            # TODO does is make sense to "outbalance" neg episode reward by scaling w t?
            if self.small_neg_rew:
                reward = -self.beta
            else:
                reward = -self.t * self.beta

        # yield reward
        return reward

    @overrides
    def forward_dynamics(self, action):
        raise NotImplementedError

    @property
    @overrides
    def action_space(self):
        low = constants.action_low
        high = constants.action_high
        return spaces.Box(low=low, high=high)

    @property
    @overrides
    def observation_space(self):
        #high = np.asarray([np.inf] * len(constants.state_keys))
        #low = -high
        return spaces.Box(low=constants.state_low[self.obs_state_ids], high=constants.state_high[self.obs_state_ids])

    @overrides
    def get_com_position(self, *com):
        return NotImplementedError
    @overrides
    def get_com_velocity(self, *com):
        return NotImplementedError

    # StandEnv functions:
    @property
    def dt(self):
        return self.timestep

    @overrides
    def action_from_keys(self, keys):
        if keys[pygame.K_LEFT]:
            return np.asarray([-1])
        elif keys[pygame.K_RIGHT]:
            return np.asarray([+1])
        else:
            return np.asarray([0])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyboard", action="store_true", help="Use keyboard as input")
    args = parser.parse_args()

    env = StandEnvVime()
    state = env.reset(height=0.45, stay=False)
    print(state)
    #_ = input("Enter")
    for _ in range(10):
        if args.keyboard:
            action = np.zeros(shape=1, dtype=np.float32)
            if keyboard.is_pressed("q"):
                action[0] = 1.0
            elif keyboard.is_pressed("a"):
                action[0] = -1.0
            action = action * 1200
        else:
            #action = env.action_space.sample()
            action = [900.0]
        state, _, _, _ = copy.deepcopy(env.step(action))
        print('state', state[5:])
    env.terminate()

if __name__ == "__main__":
    main()
