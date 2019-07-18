import numpy as np
import pygame
from rllab.envs.box2d.parser import find_body

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
                 reward_fn=goal_bias, # standEnv
                 timeout=0.02,
                 sim=False,
                 beta=1., sigma=0.25, action_penalty=32, use_proxy=False,
                 *args, **kwargs):
        super(StandEnvVime, self).__init__(
            self.model_path("stand_env.xml.mako"),
            *args, **kwargs
        )
        # Whether to simulate in rllab dynamics or use physical test stand
        self.sim = sim
        self.shutoff = False # Set this true to shutdown the test stand
        # Box2D params
        # self.obs_noise = 0 # Set to val, if additional observation noise desired
        # self.position_only = False # Set to True, if only position observation required
        self.max_weights_pos = 2
        # TODO find out what these parameters did:
        # self.goal_weights_pos = goal_weights_pos
        # self.height_bonus = height_bonus
        self.weights = find_body(self.world, "weights")
        self.initial_state = self._state # state that is copied to init Box2DEnv
        self.state = None
        #self.height_id = 5 # TODO set this id dynamically with constants.state_keys
        #self.is_normalized = True # TODO remove this param and normalized env properly
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

    def _get_heigth_rate(self, state):
        if self._prev_state is not None:
            state[state_keys.index('Height_Rate')] = state[state_keys.index('Height')] - self._prev_state[state_keys.index('Height')]
        else:
            state[state_keys.index('Height_Rate')] = 0.0

        return state

    @overrides
    def get_raw_obs(self):
        state = copy.deepcopy(self._bridge.read_state()) # returns a dict
        #print('got raw state', state)
        state['Height_Rate'] = state['Goal_Height'] = state['Goal_Velocity'] = state['Prev_Action'] = state['Prev_Reward'] = 0.
        state = self._convert_state_dict_to_np_arr(state)#np.array([state[key] for key in state_keys], dtype=np.float32)
        #print('calc raw + goal state', state)

        state = self._get_heigth_rate(state)
        #print('converted into height rate', state)

        # Convert stand env dict representation into rllab list
        self._cached_obs = state#self._convert_state_dict_to_np_arr(state)

        # Reset timestep counter
        self.t = 0

        # Goal
        self.goal_state = self.task(t=1)
        state[state_keys.index('Goal_Height')] = self.goal_state

        # TODO see if i can delete prev_state, prev_time ...
        self._prev_state = copy.deepcopy(state)
        self._prev_time = time.time()
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
            if body.userData["name"]=="weights" and not self.sim:
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
            state, _, _, _ = self.step(action, verbose=False)

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

    def close(self):
        self._bridge.close()

    # ========== Standard functions =============
    
    def step(self, action, verbose=True):
        """Perform a step of the environment"""
        #if self.is_normalized:
        #    action = action * constants.max_ma
        up = int(action[0])

        # Convert current(mA) to duty cycle and ensure its non-negative
        up_cycle = max(up, 0) / constants.max_ma
        down_cycle = max(-up, 0) / constants.max_ma

        done = self.is_current_done()

        if done: # Overwrite actions to zero actions when teststand is at the top or bottom height limit
            if self._prev_state[state_keys.index('Height')] < constants.height_min:
                action = action * 0.0 if up_cycle > 0 else action
                up_cycle = 0.0

            elif self._prev_state[state_keys.index('Height')] > constants.height_max:
                action = action * 0.0 if down_cycle > 0 else action
                down_cycle = 0.0

        self._current_time = time.time()

        # Maintain delay of 'timeout'
        if self._current_time - self._prev_time < self.timestep:
            delay = self.timestep - (self._current_time - self._prev_time)
            time.sleep(delay)
        else:
            warnings.warn("Delay exceeds set 'timeout' value")

        state = copy.deepcopy(self._bridge.read_state())
        state['Height_Rate'] = state['Goal_Height'] = state['Goal_Velocity'] = state['Prev_Action'] = state['Prev_Reward'] = 0.
        state = self._convert_state_dict_to_np_arr(state)#np.array([state[key] for key in state_keys])

        # TODO: can i move send_commands 3 lines higher and then wrap all code in get_current_obs()? 
        self._bridge.send_commands(down_cycle, up_cycle)
        self._prev_time = time.time()

        state = self._get_heigth_rate(state)

        self.t += 1
    
        # Goal
        self.goal_state = self.task(t=self.t + 1)
        state[state_keys.index('Goal_Height')] = self.goal_state

        # Goal velocity
        if self._prev_state is not None:
            state[state_keys.index('Goal_Velocity')] = state[state_keys.index('Goal_Height')] - self._prev_state[state_keys.index('Goal_Height')]
        else:
            state[state_keys.index('Goal_Velocity')] = 0.

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
            reward = -self.t * self.beta

        if verbose: print('Act: %.8f x_t+1: %.5fm, %.5fm/s, R: %.4f'%(
            action[0], state[(state_keys.index('Height'))],
            state[(state_keys.index('Height_Rate'))], reward))

        self._prev_state = copy.deepcopy(state)

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

        if not stay: # Reset test 
            self._move_test_stand_to_init(height)
        # TODO: why do I need another send default commands?
        self._send_default_commands()

        state = self.get_current_obs()
        print('state on reset', state)
        # Reset visualization
        self._write_state_to_vis(copy.deepcopy(state)) # Copies the initial state to the Box2D visualization 
        #print('after first state to vis')
        self._invalidate_state_caches() # Resets the cache of state that was built in get_current_obs; this is probably unnecessary

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
        self.close()

    #@overrides
    #def compute_reward(self, action):
    #    # Computations before stepping the world
    #    # TODO: check if I should store _prev_state here
    #    yield
    #    # Computations after stepping the world
    #    reward = self.reward_fn(self._prev_state, action, self.state)
    #    yield reward

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
        return spaces.Box(low=constants.state_low, high=constants.state_high)

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

    reward = lambda state, action, next_state: reward_fn(state, action, next_state)
    env = StandEnv(reward)
    state = env.reset(height=0.65, stay=False)
    print(state)
    _ = input("Enter")
    while True:
        if args.keyboard:
            action = np.zeros(shape=1, dtype=np.float32)
            if keyboard.is_pressed("q"):
                action[0] = 1.0
            elif keyboard.is_pressed("a"):
                action[0] = -1.0
            action = action * 1200
        else:
            action = env.action_space.sample()
        state, _, _, _ = copy.deepcopy(env.step(action))


if __name__ == "__main__":
    main()
