# Standenv
import shlex
import subprocess
import time 
import copy
import warnings
import tensorflow as tf

import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import os
os.environ["THEANO_FLAGS"] = "device=cpu"
from rllab.misc.overrides import overrides
from rllab.misc import logger

from solenoid.envs.constants import state_keys
from solenoid.envs import constants

# Connection to real test stand
from solenoid.envs.stand_bridge import StandBridge
from solenoid.envs.proxy_stand_bridge import ProxyStandBridge

# Forward dynamics for simulated test stand
from solenoid.misc import normalizers
from solenoid.forward_models import models

# For teststand tests
from solenoid.misc import tasks

class TestStand(object):

    def init_state(self, height):
        """
        Initializes the state
        """
        raise NotImplementedError
    def send_action(self, action):
        """
        Executes action and propagates state
        """
        raise NotImplementedError
    def read_state(self):
        """
        Returns the current state
        """
        raise NotImplementedError
    def close(self):
        """
        Closes the connection to the test stand
        """
        raise NotImplementedError

class TestStandSimPhysics(TestStand):
    """
    Test stand simulation, based on knowledge of test stand behavior
    """
    def __init__(self,
            env=None,
            timestep=0.02,
            data_log=None):

        self._state = None
        self.env = env # Parent node
        self.dt = timestep
        if self.env is not None:
            assert self.dt == self.env.timestep # passed timestep and environment timestep should match
        self.data_log = data_log

        # Parameters that govern forward dynamics
        # Set parameters for going up (max height to min height)
        #t_max_to_min = 2. # Time in s for test stand to go from goal_max to goal_min under max_ma (i.e., going up)
        #h_dot_max = (constants.goal_max - constants.goal_min) / t_max_to_min # Maximum height rate in s (going up)
        self.h_dot_max = 0.01 / 0.02 # (delta_height_max / timestep) # Maximum height rate in s (going up)
        self.u_lim_min =  -900. # Control in mA that results in maximum height rate
        self.u_dead_band_min = -550. # Control in mA at which starts the dead band (estimated conservatively: u_dead_band_min_est < u_dead_band_min_true)
        self.d_h_dot_d_u_up = (0. - self.h_dot_max)/(self.u_dead_band_min - self.u_lim_min) # Slope height rate over control 
        
        # Set parameters for going down (min height to max height)
        #t_min_to_max = t_max_to_min
        self.h_dot_min = - self.h_dot_max
        self.u_lim_max = - self.u_lim_min
        self.u_dead_band_max = - self.u_dead_band_min
        self.d_h_dot_d_u_down = self.d_h_dot_d_u_up

    @overrides
    def init_state(self, height=0.55,init_state_dict=None):
        logger.log('Change in Action: %.8f'%(np.mean(np.asarray(self.data_log["change_in_act"][:-2]))))
        self.data_log["change_in_act"] = [0]

        if init_state_dict is None:
            # Init state; read off from sample_rollout_0.png
            init_state_dict = {
                "Rod_Pressure": np.array([0.18]),
                "Base_Pressure": np.array([0.0]),
                "System_Pressure": np.array([0.1]), 
                "Load_Sense_Pressure": np.array([0.1]), 
                "Reservoir_Temperature": np.array([0.508]),
                "Height": np.array([height]),   
                "Height_Rate": np.array([0.]),
                "Goal_Height": np.array([0.]),
                "Goal_Velocity": np.array([0.]),
                "Prev_Action": np.array([0.]),
                "Prev_Reward": np.array([0.])
            }

        #print('init state arr', init_state_arr)
        # Convert state dict to np arr(state_dict):
        init_state_arr = np.array([init_state_dict[key][0] for key in constants.state_keys], dtype=np.float32)

        self._state = init_state_arr

    def delta_height_fn(self, action):
        """
        Computes delta height, given action (u_t), based on experienced data, similar to the depicted function
        h_dot
                    ^
        h_dot_max   |---
                    |   \
                0   |----\______----->  u_t
                    |           \
        h_dot_min   |            \___
                    |  |u_lim_min     
                    |    |u_dead_band_min
                    |       |0
                    |          |u_dead_band_max
                    |            |u_lim_max
                    |               |u_max
        Input:  action:         np.array((1,)) 
        Output: delta_height:   scalar
        # Parameters read from real test stand https://docs.google.com/document/d/16z_831MBuBFatXZmqVgQ1tiDi-yptQzAKWgK85k8Ww0/edit?usp=sharing
        """

        action = action[0]
        # Calculate height rate, given action
        if -constants.max_ma <= action < self.u_lim_min:
            h_dot = self.h_dot_max
        elif action < self.u_dead_band_min:
            h_dot = self.d_h_dot_d_u_up * (action - self.u_dead_band_min)
        elif action < self.u_dead_band_max:
            h_dot = 0.
        elif action < self.u_lim_max:
            h_dot = self.d_h_dot_d_u_down * (action - self.u_dead_band_max)
        elif action <= constants.max_ma:
            h_dot = self.h_dot_min
        else:
            raise ValueError('commanded action %f is out of range'%(action))

        # Bounds height rate, given height bounds
        if h_dot < 0. and self._state[constants.state_keys.index('Height')] <= constants.height_min:
            h_dot = 0.
        elif h_dot > 0. and self._state[constants.state_keys.index('Height')] >= constants.height_max:
            h_dot = 0.

        # Calculate delta_height
        delta_height = h_dot * self.dt
        return delta_height

    def forward_dynamics(self, state, action, done=False):
        """
        Calculate forward dynamics with assumption that all params stay static, except height
        """
        delta_state = np.zeros(state.shape)
        if not done: delta_state[constants.state_keys.index('Height')] = self.delta_height_fn(action)
        
        # TODO add noise

        return delta_state

    @overrides
    def send_action(self, action, done=False):
        """
        Simulates StandBridge.send_action
        Uses forward dynamics model to step to compute delta_state and steps the state
        Input: action as np.array((1,))
        """
        self.data_log["change_in_act"][-1] = np.abs(self.data_log["change_in_act"][-1] - action[0])
        self.data_log["change_in_act"].append(action[0])

        assert self._state is not None, "[Error] Initialize sim state via calling TestStandSimPhysics.init_state(state)"

        # Get delta prediction from physics dynamics model
        delta_state = self.forward_dynamics(self._state, action, done)

        # Compute next state
        self._state = self._state + delta_state

    @overrides
    def read_state(self):
        """
        Simulates StandBridge.read_state
        Returns states, defined in constants.state_keys until Goal_Height as dict
        """
        # Convert np arr to dict
        _state_dict = {key: self._state[constants.state_keys.index(key)] for key in constants.state_keys}
        return _state_dict

    @overrides
    def close(self):
        print('Simulation env closed')

    def tst(self):
        """
        Create a plot that shows behavior of simulation 
        """
        n_itr = 1 # Number of (stochastic) iterations
        max_time = 500 # Length of roll out in timesteps
        time_steps = np.arange(max_time) 
        start_time = time.time()
        times = np.zeros((n_itr))
        dummy_action_task = tasks.SineTask(
          steps=500, periods=2., offset=0.)
        actions = np.zeros((n_itr, max_time))
        states = {key: np.zeros((n_itr, max_time)) for key in constants.state_keys[:constants.state_keys.index("Height_Rate")+1]}

        for n_i in range(n_itr):
          # Init sim
          self.init_state(height=0.55)

          for t in time_steps:
            # Get action under policy of choice given state as input 
            action = (np.array([dummy_action_task(t=t)])-0.55)*1700.*3.
            actions[n_i, t] = action

            # Propagate forward dynamics
            self.send_action(action)
            
            # Get state
            state = self.read_state()
            print('x_t, u_t', state['Height'], action)
            for key in constants.state_keys[:constants.state_keys.index("Height_Rate")+1]:
              states[key][n_i, t] = state[key]

          # Measure time per itrs
          times[n_i] = time.time() - start_time
          start_time = time.time()

        # Plot states
        n_plts = len(constants.state_keys[:constants.state_keys.index("Height_Rate")+1]) + 1
        fig, ax = plt.subplots(nrows=n_plts, ncols=1,figsize=(20,10*n_plts), dpi=80 )

        for i, key in enumerate(constants.state_keys[:constants.state_keys.index("Height_Rate")+1]):
          #ax[i].title.set_text('averaged runs:')
          #for n_i in range(n_itr):
          #  ax[i].plot(time_steps, states[key][n_i,:], label=key+str(n_i))
          ax[i].plot(time_steps, np.mean(states[key], axis=0), label=key)
          ax[i].fill_between(time_steps, np.mean(states[key],axis=0) - np.std(states[key],axis=0),
               np.mean(states[key],axis=0) + np.std(states[key],axis=0), color='blue', alpha=0.2, label="$\pm\sigma(x_t)$")
          ax[i].legend()
          ax[i].set_ylabel(key)
          ax[i].set_xlabel('$t$ in steps of %s$s$'%(str(self.dt)))
        # Plot actions
        ax[i+1].plot(time_steps, np.mean(actions, axis=0), label="$u_t$")
        ax[i+1].fill_between(time_steps, np.mean(actions,axis=0) - np.std(actions,axis=0),
               np.mean(actions,axis=0) + np.std(actions,axis=0), color='blue', alpha=0.2, label="$\pm\sigma(u_t)$")
        ax[i+1].legend()
        ax[i+1].set_ylabel('$u_t$ in $mA$')
        ax[i+1].set_xlabel('$t$')#' in steps of %s$s$'%(str(timeout)))
        
        plt.legend()
        plt.suptitle('Simulated test stand \"physics\"-based forward dyn, averaged over %d itr, time/itr %.4fs'%(n_itr, np.mean(times)))

        plt.savefig('forw_dyn_phys_tst.png')
        #if(args.display):
        #  matplotlib.use( 'tkagg' )
        #  plt.show() 

class TestStandSim(TestStand):
    """
    Test stand simulation, based on learned forward dynamics model
    """
    def __init__(self,
            forw_model_type="gaussian",
            forw_job_dir="../../../../../solenoid/forward_models/gaussian_1000_500",
            env=None):

        self._state_tensor = None
        print('TODO: assign forward model path dynamically')
        self.forw_model_type = forw_model_type
        self.forw_job_dir = forw_job_dir
        self.env = env # Parent node

        self.init()

    def init(self):


        #print('shape' ,self.env.observation_space.shape)
        delta_space_shape = (len(constants.delta_low))
        # Create state, action, and delta normalizers.
        states_normalizer = normalizers.Pass(self.env.observation_space.shape)
        actions_normalizer = normalizers.MinMaxNormalizer(self.env.action_space.low, self.env.action_space.high)
        deltas_normalizer = normalizers.Normalizer(delta_space_shape)

        data_checkpoint = tf.train.Checkpoint(
            states_normalizer=states_normalizer,
            actions_normalizer=actions_normalizer,
            deltas_normalizer=deltas_normalizer,
        )

        # Restoring data normalizers checkpoint.
        checkpoint_path = tf.train.latest_checkpoint(os.path.join(self.forw_job_dir, "data"))
        if not checkpoint_path:
            raise FileNotFoundError("Data normalizers checkpoint not found.")
        tf.logging.info("Restoring data normalizers checkpoint from {}".format(checkpoint_path))
        data_checkpoint.restore(checkpoint_path)

        # Create the forward model and optimizer.
        self.forward_model = models.lookup[self.forw_model_type](
            states_normalizer, actions_normalizer, deltas_normalizer, is_variational=False
        )

        # create checkpoint for saving and loading
        forward_model_checkpoint = tf.train.Checkpoint(forward_model=self.forward_model)
        
        # Restore best forward model by evaluation loss.
        checkpoint_path = tf.train.latest_checkpoint(os.path.join(self.forw_job_dir, "forward_model"))
        if not checkpoint_path:
            raise FileNotFoundError("Forward model checkpoint not found.")

        tf.logging.info("Restoring forward checkpoint from {}".format(checkpoint_path))
        forward_model_checkpoint.restore(checkpoint_path)

    @overrides
    def init_state(self, height=0.55,init_state_dict=None):

        if init_state_dict is None:
            # Init state; read off from sample_rollout_0.png
            init_state_dict = {
                "Rod_Pressure": np.array([0.18]),
                "Base_Pressure": np.array([0.0]),
                "System_Pressure": np.array([0.1]), 
                "Load_Sense_Pressure": np.array([0.1]), 
                "Reservoir_Temperature": np.array([0.508]),
                "Height": np.array([height]),   
                "Height_Rate": np.array([0.]),
                "Goal_Height": np.array([0.]),
                "Goal_Velocity": np.array([0.]),
                "Prev_Action": np.array([0.]),
                "Prev_Reward": np.array([0.])
            }

        #print('init state arr', init_state_arr)
        # Convert state dict to np arr(state_dict):
        init_state_arr = np.array([init_state_dict[key][0] for key in constants.state_keys[:constants.state_keys.index("Height_Rate")+1]], dtype=np.float32)

        # Convert np arr to tf tensor and shape episode,step,state_dim
        tensor = tf.convert_to_tensor(init_state_arr, dtype=tf.float32)
        self._state_tensor = tf.expand_dims(tf.expand_dims(tensor,axis=0,),axis=0,)

        self.reset_state = True # Status flag whether to reset hidden state of rnn forw dyn model

    @overrides
    def send_action(self, action, done=False):
        """
        Simulates StandBridge.send_action
        Uses forward dynamics model to step to compute delta_state and steps the state
        Input: action as np.array((1,))
        """

        assert self._state_tensor is not None, "[Error] Initialize sim state via calling TestStandSim.init_state(state)"
        
        # Convert action to tensor and shape episode,steps,action_dim
        action_tensor = tf.expand_dims(
            tf.expand_dims(tf.convert_to_tensor(action, dtype=tf.float32), axis=0), axis=0
        )
        if done: self.reset_state = True # reset hidden state of RNN forw dyn model
        # Get delta prediction from trained dynamics model
        delta = self.forward_model.step(
            self._state_tensor, action_tensor, training=False,  mode=False, reset_state=self.reset_state,#step_count == 0,
        )
        self.reset_state = False

        # Compute next state
        self._state_tensor = self._state_tensor + delta

    @overrides
    def read_state(self):
        """
        Simulates StandBridge.read_state
        Returns states, defined in constants.state_keys until Goal_Height as dict
        """
        # TODO remove with sess.  as default
        assert tf.executing_eagerly(), "Not in tf eager mode. Call tf.enable_eager_execution() at start of code-worker (e.g., in run_experiment)"
        # Convert tf tensor to np array
        _state = np.zeros(len(constants.state_keys))
        _state[:constants.state_keys.index("Height_Rate")] = np.squeeze(self._state_tensor[..., :-1].numpy())
        # Convert np arr to dict
        #print('_st', _state)
        _state = {key: _state[constants.state_keys.index(key)] for key in constants.state_keys}
        return _state

    @overrides
    def close(self):
        print('Simulation env closed')

class TestStandReal(TestStand):
    def __init__(self,
            use_proxy,
            env,
            timestep=0.02,
            tst=False
        ):
        self._use_proxy = use_proxy
        self.env = env
        self.timestep = timestep
        # TODO eliminate this test flag
        if not tst:
            assert self.timestep == self.env.timestep # passed timestep and environment timestep should match
        # Init bridge to test stand
        if self._use_proxy:
            self._bridge = ProxyStandBridge()
        else:
            # Create Bridge to physical test stand
            find_usbid_cmd = shlex.split("ls /dev/serial/by-id/")
            usb_id = "/dev/serial/by-id/" + subprocess.check_output(find_usbid_cmd).decode()
            usb_id = usb_id[0:-1]
            self._bridge = StandBridge(port=usb_id)

        # LQR control to reset test stand to a Prespecified Height
        self.K = constants.K

    def _send_default_commands(self):
        """
        Send idle commands to test stand
        """
        self._bridge.send_default_commands()
        time.sleep(self.env.timestep)

    @overrides
    def init_state(self, height=0.55):
        """
        Moves test stand to initial position
        """
        self._send_default_commands()
        state = self.env.get_raw_obs()
        if self._use_proxy:
            state[state_keys.index('Height')] = height

        max_init_steps = 100
        i = 0
        while (100*state[state_keys.index('Height')]).astype(int) != np.array([100*height]).astype(int) and i < max_init_steps:
            # Compute error in current height and desired setpoint
            state[state_keys.index('Height')] = state[state_keys.index('Height')] - height
            # Drive error in height to zero
            # TODO: Replace this LQR K with eliminated-dead-bands K
            action = np.dot(np.negative(self.K), np.array(state[:state_keys.index("Height_Rate")+1]))
            state, _, _, _ = self.env.step(action, partial_obs='full')
            i += 1
        # Send default commands to stop test stand from moving
        self._send_default_commands()

    @overrides
    def send_action(self, action, done):
        """
        Sends action to test stand and maintains constant rate
        """
        # TODO keep track of time and state in testStandReal class, not env

        # Convert action from current(mA) to duty cycle and ensure its non-negative
        up = int(action[0])
        up_cycle = max(up, 0) / constants.max_ma
        down_cycle = max(-up, 0) / constants.max_ma

        # Overwrite actions to zero actions when teststand is at the top or bottom height limit
        if done: 
            if self.env._prev_state[state_keys.index('Height')] < constants.height_min:
                action = action * 0.0 if up_cycle > 0 else action
                up_cycle = 0.0

            elif self.env._prev_state[state_keys.index('Height')] > constants.height_max:
                action = action * 0.0 if down_cycle > 0 else action
                down_cycle = 0.0

        self.env._current_time = time.time()

        # Maintain delay of 'timeout'
        if self.env._current_time - self.env._prev_time < self.env.timestep:
            delay = self.env.timestep - (self.env._current_time - self.env._prev_time)
            time.sleep(delay)
        else:
            warnings.warn("Delay exceeds set 'timeout' value")
            print("[WARNING] Delay exceeds set 'timeout' value")

        # TODO: write try catch for sending commands
        self._bridge.send_commands(down_cycle, up_cycle)

        self.env._prev_time = time.time()

    @overrides
    def read_state(self):
        _state_dict = {key: np.zeros((1,)) for key in constants.state_keys}
        state_bridge = self._bridge.read_state()
        for key in state_bridge.keys():
            _state_dict[key] = state_bridge[key] 
        return _state_dict

    @overrides
    def close(self):
        self._bridge.close()

    def tst(self):
        # Drive test stand down
        action = [-1700]
        up = int(action[0])
        up_cycle = max(up, 0) / constants.max_ma
        down_cycle = max(-up, 0) / constants.max_ma
        self._bridge.send_commands(down_cycle, up_cycle)
        self._bridge.send_commands(down_cycle, up_cycle)
    
if __name__=="__main__":
    #testStand = TestStandSimPhysics(timestep=0.02)
    testStand = TestStandReal(use_proxy=False, env=None, timestep=0.02, tst=True)
    #testStand._send_default_commands()
    testStand.tst()