import argparse
import copy
import subprocess
import shlex
import time
import warnings

import gym
from gym import spaces
import keyboard
import numpy as np

from envs import constants
from envs.stand_bridge import StandBridge


class StandEnv(object):
    def __init__(self, reward_fn, timeout=0.04):
        """
        Args:
        reward_fn : A function which returns rewards for the solenoid environment.
                    Input to the function is state, action, next_state and output is
                    a scalar reward value (float)

        timeout (float) : Ensures a consistent delay of 'timeout' seconds b/w action,state and next state

                %%%%%%% --> Piston
                %%%%%%%          
                (###(          
                 ###  --> Top permitted Height = 0.3      
                 ###           
             .(((((((((.       
             ((((((((((( --> Weights (1100 pounds)      
             ((#(((((#((       
             */(((((((/*      
                .%#%.          
                 ###           
                 ###           
                 ###           
                 ###          
                 ###   --> Bottom permitted Height = 0.8        
                 ###           
              =========
              =========
        
        ===========             ================        
        State                   Bounds              
        ===========             ================        
        Height                   [0.3, 0.8]                                                      
        Height Rate              [-0.03, +0.03] - estimate
        Reservoir Temperature    [0.35, 0.53] - estimate 
        Rod Pressure             [0.0, 0.9] - estimate                 
        Base Pressure            [0.0, 0.35] - estimate
        System Pressure          [0.0, 0.9] - estimate
        Load Sense Pressure      [0.0, 0.9] - estimate          
        ===========              ================  
        """

        self._timeout = timeout

        self.reward_fn = reward_fn

        # TODO : Find actual state high and low boundaries
        high = np.asarray([np.inf] * len(constants.state_keys))
        low = -high
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        low = constants.action_low
        high = constants.action_high
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        find_usbid_cmd = shlex.split("ls /dev/serial/by-id/")
        usb_id = "/dev/serial/by-id/" + subprocess.check_output(find_usbid_cmd).decode()
        usb_id = usb_id[0:-1]
        self._bridge = StandBridge(port=usb_id)

        self._prev_state = None

        # LQR control for reset to a Prespecified Height
        self.K = constants.K

    @property
    def dt(self):
        return self._timeout

    def _reset(self):
        """Send zero actions and reset teststand to same height as before"""

        self._bridge.send_default_commands()
        time.sleep(self._timeout)
        state = copy.deepcopy(self._bridge.read_state())
        if self._prev_state is not None:
            state["Height_Rate"] = state["Height"] - self._prev_state["Height"]
        else:
            state["Height_Rate"] = 0.0
        self._prev_state = copy.deepcopy(state)
        self._prev_time = time.time()
        return state

    def reset(self, height=0.55, stay=False):
        """
        Reset Teststand Environment to a specific height or to the previous height

        Args:

        height (float) : specify height to which teststand must be reset
        
        stay (bool): If True, send zero actions and reset to same height as earlier position
                     If False, reset to height specified by param height
        """
        if stay:
            state = self._reset()
        else:
            state = self._reset()
            for i in range(100):
                # Compute error in current height and desired setpoint
                state["Height"] = state["Height"] - height
                # Drive error in height to zero
                state_list = []
                for key in range(len(constants.state_keys)):
                    state_list.append(state[constants.state_keys[key]])
                action = np.dot(np.negative(self.K), np.array(state_list))
                state, _, _, _ = self.step(action)
            state = self._reset()
        return state

    def step(self, action):
        """Perform a step of the environment"""

        up = int(action[0])

        # Convert current(mA) to duty cycle and ensure its non-negative
        up_cycle = max(up, 0) / constants.max_ma
        down_cycle = max(-up, 0) / constants.max_ma

        done = False

        # Send zero actions when teststand is at the top or bottom height limit
        if self._prev_state["Height"] < constants.height_min:
            action = action * 0.0 if up_cycle > 0 else action
            up_cycle = 0.0
            # Episode terminates if teststand crosses the height limits
            done = True

        elif self._prev_state["Height"] > constants.height_max:
            action = action * 0.0 if down_cycle > 0 else action
            down_cycle = 0.0
            # Episode terminates if teststand crosses the height limits
            done = True

        self._current_time = time.time()

        # Maintain delay of 'timeout'
        if self._current_time - self._prev_time < self._timeout:
            delay = self._timeout - (self._current_time - self._prev_time)
            time.sleep(delay)
        else:
            warnings.warn("Delay exceeds set 'timeout' value")

        state = copy.deepcopy(self._bridge.read_state())
        self._bridge.send_commands(down_cycle, up_cycle)

        self._prev_time = time.time()

        # Compute Height_Dot
        if self._prev_state is not None:
            state["Height_Rate"] = state["Height"] - self._prev_state["Height"]
        else:
            state["Height_Rate"] = 0.0

        reward = self.reward_fn(self._prev_state, action, state)

        self._prev_state = copy.deepcopy(state)

        info = {}

        return state, reward, done, info

    def close(self):
        self._bridge.close()


def reward_fn(state, action, next_state):
    # Stand-in reward function - replace with trajectory specific reward
    return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyboard", action="store_true", help="Use keyboard as input")
    args = parser.parse_args()

    reward = lambda state, action, next_state: reward_fn(state, action, next_state)
    env = StandEnv(reward)
    state = env.reset(height=0.65, stay=False)
    env.close()
    return 
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
            print(action)
        state, _, _, _ = copy.deepcopy(env.step(action))
if __name__ == "__main__":
    main()
