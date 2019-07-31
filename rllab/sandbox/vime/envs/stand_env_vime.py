import numpy as np
import pygame
from rllab.envs.box2d.parser import find_body
import argparse
import os
os.environ["THEANO_FLAGS"] = "device=cpu"
import matplotlib.pyplot as plt

# Standenv
import time 
import copy

from tensorboardX import SummaryWriter

from rllab import spaces
from rllab.core.serializable import Serializable
from rllab.envs.box2d.box2d_env import Box2DEnv
from rllab.misc import autoargs
from rllab.misc.overrides import overrides

from solenoid.envs.constants import state_keys
from solenoid.envs import constants
from solenoid.misc import tasks
from solenoid.misc.reward_fns import goal_bias

from rllab.sandbox.vime.envs.test_stand import TestStandSim, TestStandSimPhysics, TestStandReal

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
                 sim=False,
                 beta=1., sigma=0.05, action_penalty=4, use_proxy=False,
                 small_neg_rew=False,
                 partial_obs=None,
                 t_lookahead=0,
                 t_past=0,
                 init_w_lqt=False,
                 dead_band=0.,
                 max_action=None,
                 verbose=True,
                 log_tb=True,
                 *args, **kwargs):
        super(StandEnvVime, self).__init__(
            self.model_path("stand_env.xml.mako"),
            *args, **kwargs
        )
        #print('in eager mode at standenv init', tf.executing_eagerly())
        #tf.enable_eager_execution()

        self.vis = vis # if true, visualize in rllab env 
        self.shutoff = False # Set this true to shutdown the test stand
        self.verbose = verbose # if true, print logs

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
        self.dead_band = dead_band # Scale action space to {[-max_ma, -dead_b],[dead_b, max_ma]}
        self.max_action = max_action

        # Define partial observation 
        if self.partial_obs=='height_only':# Set obs to [height_t, goal_height] 
            self.obs_state_ids = [state_keys.index('Height'),state_keys.index('Goal_Height')]
        elif self.partial_obs=='err_only':# Set obs to [goal_height - height_t]
            self.obs_state_ids = [state_keys.index('Err')]
        else: # TODO only take in state until goal height!
            self.obs_state_ids = range(len(state_keys))

        # StandEnv
        if task is None:
            task = tasks.SineTask(
                steps=500, periods=2., offset=0.)

        # Define goal/reward fn
        self.task = task
        self.reward_fn = reward_fn
        self.beta = beta
        self.sigma = sigma
        self.action_penalty = action_penalty

        # TensorboardX
        if log_tb: self.writer = SummaryWriter(logdir="runs/tst")#logdir="runs/tst", comment="tst2", filename_suffix="_suffix_tst")

        # Log 
        self.data_log = {
            "change_in_act": [0],
            "reward": []
         }


        # Keep track of timestep.
        self.t = 0
        self.timestep = timeout # overwrites box2D timestep
        self.n_i = 0 # number of iteration during training

        # Init test stand
        self.sim = sim # if true step in forward dyn simulation; if false step on physical test stand
        if self.sim=="sim":
            self.test_stand = TestStandSim(env=self)
        elif self.sim=="sim_physics":
            self.test_stand = TestStandSimPhysics(env=self, timestep=self.timestep, data_log=self.data_log)
        elif self.sim=="real":
            self.test_stand = TestStandReal(env=self, use_proxy=use_proxy)
        
        self._prev_time = time.time() # Used to maintain fixed rate of action commands on physical test stand
        self._prev_state = None

        # Init Box2D serializable
        Serializable.quick_init(self, locals())

    # ======== Process state reading from test stand ==============
    def _convert_state_dict_to_np_arr(self, state_dict):
        return np.array([state_dict[key] for key in state_keys], dtype=np.float32)

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
        # Get state from Test stand
        state = copy.deepcopy(self.test_stand.read_state()) # returns a dict

        state['Height_Rate'] = state['Goal_Height'] = state['Goal_Velocity'] = state['Prev_Action'] = state['Prev_Reward'] #= state['Err'] = 0.
        state = self._convert_state_dict_to_np_arr(state)

        if 'Height_Rate' in state_keys: state = self._get_height_rate(state)  


        # Goal
        self.goal_state = self.task(t=self.t + 1)
        state[state_keys.index('Goal_Height')] = self.goal_state

        # Goal velocity
        #if self._prev_state is not None:
        state[state_keys.index('Goal_Velocity')] = state[state_keys.index('Goal_Height')] - self.task(t=self.t)#self._prev_state[state_keys.index('Goal_Height')]
        #else:
        #    state[state_keys.index('Goal_Velocity')] = 0.

        # Error
        if 'Err' in state_keys: state[state_keys.index('Err')] = state[state_keys.index('Goal_Height')] - state[state_keys.index('Height')]

        # write state to visualization
        if self.vis: self._write_state_to_vis(copy.deepcopy(state)) # Copies the initial state to the Box2D visualization 

        # Making sure that prev state contains goal from one step before, s.t. reward function penalizes correct height, goal pairs
        self._prev_state = copy.deepcopy(state)
        self._prev_state[state_keys.index('Goal_Height')] = self.task(t=self.t)

        # Iterate timestep counter
        self.t += 1

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
            #print('body user data', body.userData)
            if body.userData["name"]=="weights":# and not self.vis:
                # TODO This is very hard-coded. update this if I change visualization
                #print('forcing state onto vis: height, heigh rate', state[state_keys.index('Height')], state[state_keys.index('Height_Rate')])
                s.append(np.concatenate([
                    list((0., state[state_keys.index('Height')])),
                    [body.angle],
                    list((0., state[state_keys.index('Height_Rate')])),
                    [body.angularVelocity]
                ]))
            else:
                s.append(np.concatenate([
                    list(body.position),
                    [body.angle],
                    list(body.linearVelocity),
                    [body.angularVelocity]
                ]))
        state = np.concatenate(s)
        splitted = np.array(state).reshape((-1, 6))
        for body, body_state in zip(self.world.bodies, splitted):
            #print('setting state for body: with state:', body, body_state)
            xpos, ypos, apos, xvel, yvel, avel = body_state
            body.position = (xpos, ypos)
            body.angle = apos
            body.linearVelocity = (xvel, yvel)
            body.angularVelocity = avel

        return np.concatenate(s)

    # ========== Standard functions =============
    def step(self, action, partial_obs=None):
        """Perform a step of the environment"""
        # TODO evaluate why done is called before state transition?
        done = self.is_current_done()

        # Rescale dead-band
        if action >= 0.:
            action += self.dead_band
        elif action < 0.:
            action -= self.dead_band

        self.test_stand.send_action(action, done=done)

        state = self.get_current_obs(partial_obs)

        reward = self.compute_reward(action, done)

        if self.verbose and self.partial_obs=='height_only': print('t: %3d, u_t: %.8f x_t: %.5fm, g_t+1: %.5fm, R: %.4f'%(
            self.t, action[0], state[0],state[1], reward))
        elif self.verbose and self.partial_obs=='err_only': print('t: %3d, u_t: %.8f err_t: %.5fm, R: %.4f'%(
            self.t, action[0], state[0], reward))
        elif self.verbose: print('t: %4d u_t: %14.8f x_t: %7.5fm, %8.5fm/s, x_g_t: %7.5f , %11.8fm/s, R: %9.4f'%(
            self.t,
            action[0], state[(state_keys.index('Height'))],
            state[(state_keys.index('Height_Rate'))],
            state[(state_keys.index('Goal_Height'))],
            state[(state_keys.index('Goal_Velocity'))],
            reward))

        info = {}

        if self.vis: self.render()

        if self.writer:
            self.data_log["reward"].append(reward)

        return state, reward, done, info


    @overrides
    def reset(self, height=0.55, stay=False):
        """
        Reset test stand and visualization. Environment to a specific height or to the previous height

        Args:

        height (float) : specify height to which teststand must be reset
        
        stay (bool): If True, send zero actions and reset to same height as earlier position
                     If False, reset to height specified by param height
        """
        print('writing:', self.writer, np.mean(np.asarray(self.data_log["change_in_act"][:-2])), self.n_i, np.mean(np.asarray(self.data_log["reward"])))
        if self.writer:
            self.writer.add_scalar('data/change_in_act', np.mean(np.asarray(self.data_log["change_in_act"][:-2])), self.n_i)
            self.writer.add_scalar('data/reward', np.mean(np.asarray(self.data_log["reward"])), self.n_i)
        self.data_log["reward"] = []

        print('RESET CALLED to height: ', height)
        self._prev_time = time.time()
        if not stay: # Reset test 
            self.test_stand.init_state(height=height)

        # Reset timestep counter
        self.t = 0
        state = self.get_current_obs()
        if self.verbose:
            print('state after reset: t: %4d x_t: %7.5fm, %8.5fm/s, x_g_t: %7.5f , %11.8fm/s'%(
                self.t, state[(state_keys.index('Height'))],
                state[(state_keys.index('Height_Rate'))],
                state[(state_keys.index('Goal_Height'))],
                state[(state_keys.index('Goal_Velocity'))]
                ))
        
        # Iterate number of training iteration
        self.n_i += 1
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
        print('reset')
        self.test_stand.close()

    @overrides
    def compute_reward(self, action, done):
        # Computations before stepping the world
        # TODO: check if I should store _prev_state here
        # TODO check if i should write yield instead of return
        # yield
        # Computations after stepping the world

        # TODO: evaluate why reward fn takes in prev_state, not state
        #print('comp rew', action, done)
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
        # TODO link this fn 
        raise NotImplementedError

    @property
    @overrides
    def action_space(self):
        if self.max_action:
            constants.action_high = np.array([self.max_action], dtype=np.float32)
            constants.action_low = -constants.action_high    
        high = constants.action_high - self.dead_band 
        low = constants.action_low + self.dead_band

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

def plot_states(time_steps, n_itr, states, actions, times, rewards,args):

    # Plot states
    n_plts = len(constants.state_keys[:constants.state_keys.index('Height_Rate')]) + 1
    n_plts = 3
    fig, ax = plt.subplots(nrows=n_plts, ncols=1,figsize=(20,10*n_plts), dpi=80 )

    for i, key in enumerate(constants.state_keys[constants.state_keys.index('Height'):constants.state_keys.index('Height_Rate')]):
      #ax[i].title.set_text('averaged runs:')
      #for n_i in range(n_itr):
      #  ax[i].plot(time_steps, states[key][n_i,:], label=key+str(n_i))
      ax[i].plot(time_steps, np.mean(states[key], axis=0), label=key)
      ax[i].fill_between(time_steps, np.mean(states[key],axis=0) - np.std(states[key],axis=0),
           np.mean(states[key],axis=0) + np.std(states[key],axis=0), color='blue', alpha=0.2, label="$\pm\sigma(x_t)$")
      #ax[i].plot(time_steps, goals[b_i,:], label="$x_{des}$")
      ax[i].legend()
      ax[i].set_ylabel(key)
      ax[i].set_xlabel('$t$')#' in steps of %s$s$'%(str(timeout)))
      if key == "Height": # Plot Goal height into height plot
          key = "Goal_Height"
          ax[i].plot(time_steps, np.mean(states[key], axis=0), label=key)
          ax[i].legend()

    # Plot actions
    ax[i+1].plot(time_steps, np.mean(actions, axis=0), label="$u_t$")
    ax[i+1].fill_between(time_steps, np.mean(actions,axis=0) - np.std(actions,axis=0),
           np.mean(actions,axis=0) + np.std(actions,axis=0), color='blue', alpha=0.2, label="$\pm\sigma(u_t)$")
    #print('goal vel', states["Goal_Velocity"][0, :5])
    #ax[i+1].plot(time_steps, np.mean(states["Goal_Velocity"], axis=0), label="Goal_Velocity")
    ax[i+1].legend()
    ax[i+1].set_ylabel('$u_t$ in $mA$')
    ax[i+1].set_xlabel('$t$')#' in steps of %s$s$'%(str(timeout)))

    # Plot rewards
    ax[i+2].title.set_text('avg Rew: '+ str(np.mean(np.mean(rewards,axis=0))))
    ax[i+2].get_yaxis().get_major_formatter().set_useOffset(False)
    ax[i+2].plot(time_steps, np.mean(rewards, axis=0), label="$R_t$")
    ax[i+2].fill_between(time_steps, np.mean(rewards,axis=0) - np.std(rewards,axis=0),
           np.mean(rewards,axis=0) + np.std(rewards,axis=0), color='blue', alpha=0.2, label="$\pm\sigma(R_t)$")
    ax[i+2].legend()
    ax[i+2].set_ylabel('$R$')
    ax[i+2].set_xlabel('$t$')#' in steps of %s$s$'%(str(timeout)))
    
    plt.legend()
    plt.suptitle('Simulated forward dyn on tst stand, averaged over %d itr, time/itr %.4fs'%(n_itr, np.mean(times))) # raised title

    plt.savefig('forw_dyn_tst.png')
    if(args.display):
      matplotlib.use( 'tkagg' )
      plt.show() 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyboard", action="store_true", help="Use keyboard as input")
    parser.add_argument('--sim', type=str, default="sim_physics",
                        choices=['sim','sim_physics','real'], help='Name of teststand to run on')
    parser.add_argument("--display", action="store_true", help="Display plot")
    args = parser.parse_args()

    dead_band=0.#550.
    max_action = 900.
    timestep = 0.02
    env = StandEnvVime(sim=args.sim,
        dead_band=dead_band,
        max_action=max_action,
        timeout=timestep)
    state = env.reset(height=0.55, stay=False)
    print(state)

    n_itr = 5
    max_time = 500
    time_steps = np.arange(max_time)
    times = np.zeros((n_itr))
    start_time = time.time()
    # For plotting
    actions = np.zeros((n_itr, max_time))
    rewards = np.zeros((n_itr, max_time))
    states = {
      "Rod_Pressure": np.zeros((n_itr, max_time)),
      "Base_Pressure": np.zeros((n_itr, max_time)),
      "System_Pressure": np.zeros((n_itr, max_time)), 
      "Load_Sense_Pressure": np.zeros((n_itr, max_time)), 
      "Reservoir_Temperature": np.zeros((n_itr, max_time)),
      "Height": np.zeros((n_itr, max_time)),
      "Height_Rate": np.zeros((n_itr, max_time)),
      "Goal_Height": np.zeros((n_itr, max_time)),
      "Goal_Velocity": np.zeros((n_itr, max_time)),
      "Prev_Action": np.zeros((n_itr, max_time)),
      "Prev_Reward": np.zeros((n_itr, max_time))
    }
    d_h_dot_d_u_up = -1./700.
    periods=2.
    offset = 0. + np.pi/2. # offset from goal and cosinus
    t_max = 500.
    u_dead_band_min = -550.
    task = tasks.SineTask(
        steps=t_max, periods=periods, offset=offset, shift_and_scale=False)

    #_ = input("Enter")
    for n_i in range(n_itr):
        env.reset() 
        for t in range(max_time):
            if args.keyboard:
                action = np.zeros(shape=1, dtype=np.float32)
                if keyboard.is_pressed("q"):
                    action[0] = 1.0
                elif keyboard.is_pressed("a"):
                    action[0] = -1.0
                action = action * 1200
            else:
                #action = env.action_space.sample()
                
                delta_goal = np.array([task(t=t+1)], dtype=np.float64) * 2. * np.pi * periods / t_max  # Derivative of goal task : sin(h_offset + 2pi n_periods t/t_max)
                delta_goal_height = delta_goal * 1./2. * (constants.goal_max - constants.goal_min)  # Derivative of shift and scale of goal task
                action_incr = 1./d_h_dot_d_u_up * delta_goal_height 
                action = 1./timestep * action_incr # 1/dt * action
                if action <= 0.:
                    action += u_dead_band_min
                elif action > 0.:
                    action -= u_dead_band_min
            state, reward, done, info = copy.deepcopy(env.step(action))

            # For plotting
            actions[n_i,t] = action[0]
            for key in constants.state_keys:
                states[key][n_i, t] = state[constants.state_keys.index(key)]
            rewards[n_i, t] = reward

            print('reward, done, info', reward, done, info)
        # Measure time per itrs
        times[n_i] = time.time() - start_time
        start_time = time.time()
        print('n_i#', n_i)
    env.terminate()

    plot_states(time_steps, n_itr, states, actions, times, rewards, args)

if __name__ == "__main__":
    main()
