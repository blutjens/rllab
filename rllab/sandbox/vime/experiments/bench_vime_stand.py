import argparse

import joblib
import tensorflow as tf

from rllab.misc.console import query_yes_no
from rllab.sampler.utils import rollout

import os
os.environ["THEANO_FLAGS"] = "device=cpu"
import numpy as np
import matplotlib
import matplotlib.pyplot as plt#
#from brokenaxes import brokenaxes
from matplotlib.gridspec import GridSpec
        
from solenoid.envs.constants import state_keys
from solenoid.envs import constants

from solenoid.misc.metrics import mean_cumulative_absolute_error, mean_cumulative_squared_error
from solenoid.misc.tasks import SineTask,ChirpTask,StepTask#,ToZeroTask

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='logs_vime_stand_4_0_res/itr_17',
                        help='path to the snapshot file')
    parser.add_argument('--plt_title', type=str, default='',
                        help='plot title')
    parser.add_argument('--max_path_length', type=int, default=500,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=1,
                        help='Speedup')
    parser.add_argument('--display', action="store_true", default=False,
                        help='plot and print results')
    parser.add_argument('--not_save_fig', action="store_false", default=False,
                        help='save plot in file')
    parser.add_argument('--deterministic', action="store_true", default=False,
                        help='rollout deterministic')
    #parser.add_argument('--tst_scenario', type=str, default=None,
    #                    #choices=['chirp', 'step', 'larger_deadbands', 'no_deadbands', 
    #                    #'lower_height_rate_up', 'greater_height_rate_up', 'learn_lqt_plus_rl', 'real'],
    #                    help='test scenario')
    parser.add_argument('--tst_scenario', action="store", nargs='*', type=str, default=None,
                        choices=['sine', 'chirp', 'step', 'larger_deadbands', 'no_deadbands', 
                        'lower_height_rate_up', 'greater_height_rate_up', 'learn_lqt_plus_rl', 'real'],
                        help='test scenario')
    args = parser.parse_args()

    model_file = 'data/' + args.model + '.pkl'
    # If the snapshot model_file use tensorflow, do:
    # import tensorflow as tf
    # with tf.Session():
    #     [rest of the code]y
    #with tf.Session() as sess:
    data = joblib.load(model_file)
    policy = data['policy']
    env = data['env']
    env.log_tb = False # Don't create tensorboard logs during benchmark
    if args.display:
        print('env', env)
        print('settin gvis true')
        env.wrapped_env.vis = True
    else:
        print('not disp')
        env.wrapped_env.vis = False
    n_bench_itr = 2
    tst_params = ['sine']#, 'chirp', 'step']
    plot_mult_tst_params = True
    plt.rcParams.update({'font.size': 28})
    
    n_plts_p_param = 6
    fig_height = len(tst_params)*10*n_plts_p_param 
    fig = plt.figure(figsize=(20,fig_height), dpi=80 )
    sps = GridSpec(nrows=len(tst_params)*n_plts_p_param, ncols=1)    #fig = plt.figure(figsize=(5,5))

    for p_i, tst_param in enumerate(tst_params):

        # Get env/net/policy params for printing
        algo_name_long = data['algo'].__class__.__module__ + '.' + data['algo'].__class__.__name__
        vime = '+ VIME' if 'vime' in algo_name_long else ''
        trpo = 'TRPO' if 'trpo' in algo_name_long else ''
        algo_name = trpo + vime
        step_size = data['algo'].step_size
        #if vime != '':
        #    r_train = None#np.mean(data['episode_rewards'])
        #    eps_length = int(np.mean(data['episode_lengths']))
        #else:  
        r_train = None
        eps_length = int(data['algo'].max_path_length)
        train_itr_res = data['itr']
        net_in_dim = policy._mean_network._layers[0].shape[1]
        net_shape = [layer.num_units for layer in policy._mean_network._layers[1:-1]]
        net_out_dim = policy._mean_network._layers[-1].num_units
        beta = env.wrapped_env.beta
        sigma = env.wrapped_env.sigma
        rew_fn = env.wrapped_env.reward_fn.__name__
        task = env.wrapped_env.task.__class__.__name__
        dead_band = env.wrapped_env.dead_band 

        env.wrapped_env.sigma = 0.05
        print('set eval reward fn sigma to ', env.wrapped_env.sigma)

        # Test under different task / goal / transition dyn
        if "sine" in args.tst_scenario:
            env.wrapped_env.task = SineTask(
                steps=500, periods=1., offset=0.)    
        if "chirp" in args.tst_scenario:
            env.wrapped_env.task = ChirpTask(
                    steps=500, periods=1., offset=0.)
        if "step" in args.tst_scenario:
            env.wrapped_env.task = StepTask(
                    steps=500, periods=1., offset=0.)    
        if "larger_deadbands" in args.tst_scenario:
            env.wrapped_env.test_stand.u_dead_band_min = -750.
            env.wrapped_env.test_stand.u_dead_band_max = -env.wrapped_env.test_stand.u_dead_band_min
        if "no_deadbands" in args.tst_scenario:
            env.wrapped_env.test_stand.u_dead_band_min = -0.1
            env.wrapped_env.test_stand.u_dead_band_max = -env.wrapped_env.test_stand.u_dead_band_min
        if "lower_height_rate_up" in args.tst_scenario:
            env.wrapped_env.test_stand.d_h_dot_d_u_up = -1./1300.
            env.wrapped_env.test_stand.d_h_dot_d_u_down = -1./100.#50.
        if "greater_height_rate_up" in args.tst_scenario:
            env.wrapped_env.test_stand.d_h_dot_d_u_up = -1./100.
            env.wrapped_env.test_stand.d_h_dot_d_u_down = -1./1500.
        # Add RL on top of LQT controller 
        if "learn_lqt_plus_rl" in args.tst_scenario:      
            env.wrapped_env.learn_lqt_plus_rl = True # If true, do u = LQT(x) + RL(x) 
            env.wrapped_env.lqt_t_lookahead = 5 # Lookahaed time of LQT 
            from solenoid.controls.lqt import LQT
            # TODO set this path with controls module path.
            dynamics_path = '../../../../../solenoid/controls/data/teststand_AB_opt_on_sim.pkl'
            env.wrapped_env.lqt = LQT(dynamics_path)
        # Test on real teststand
        if "real" in args.tst_scenario:
            from rllab.sandbox.vime.envs.test_stand import TestStandReal
            env.wrapped_env.sim = "real"
            env.wrapped_env.test_stand = TestStandReal(env=env.wrapped_env, use_proxy=env.wrapped_env.use_proxy)


        #env.wrapped_env.task = SineTask(
        #        steps=500, periods=1., offset=0.)

        mcae_scores = np.zeros(n_bench_itr)
        cum_rews = np.zeros(n_bench_itr)
        mcse_scores = np.zeros(n_bench_itr)
        heights = np.zeros((n_bench_itr, args.max_path_length))
        goals = np.zeros((n_bench_itr, args.max_path_length))
        actions = np.zeros((n_bench_itr, args.max_path_length))
        rewards = np.zeros((n_bench_itr, args.max_path_length))
        change_in_actions = np.zeros((n_bench_itr))
        act_std_dev = np.zeros((args.max_path_length-1)) 
        for b_i in range(n_bench_itr):

            # Execute and log 
            path = rollout(env, policy, max_path_length=args.max_path_length,
                           animated=False, speedup=args.speedup)

            # Read states
            print('path keys', path.keys())
            state = {key:path['observations'][:,state_keys.index(key)] for key in state_keys}
            print('state h',  state['Height'].shape)
            heights[b_i,:len(state['Height'])] = state['Height']
            goals[b_i,:len(state['Goal_Height'])] = state['Goal_Height']
            actions[b_i,:len(path['actions'][:,0])] = path['actions'][:,0]
            rewards[b_i,:len(path['rewards'])] = path['rewards']
            cum_rews[b_i] = np.sum(np.array(rewards[b_i,:]))
            time_steps = np.arange(state['Height'].shape[0])#np.zeros((args.max_path_length))#
            # Get env params
            print('ts', time_steps.shape)
            print('rew', rewards.shape)
            timeout = env.wrapped_env.timestep

            # Scale actions to [-max_ma, max_ma]
            lb, ub = env.wrapped_env.action_space.bounds
            print('action bnd', lb, ub)
            print('actions', actions[b_i,:10])
            actions[b_i,:] = lb + (actions[b_i,:] + 1.) * 0.5 * (ub - lb)
            actions[b_i,:] = np.clip(actions[b_i,:], lb, ub)
            print('actions', actions[b_i, :10])
            # Calculate change and std dev in actions
            change_in_actions[b_i] = np.mean(np.abs(actions[b_i,1:] - actions[b_i,:-1]))

            #Convert to shape [episodes *time * number of height dimensions] for metric computation
            achieved_heights = np.expand_dims(np.expand_dims(np.array(heights[b_i,:]).T,axis=0),axis=2)
            target_heights = np.expand_dims(np.expand_dims(np.array(goals[b_i,:]).T, axis = 0),axis=2)
            print('ach h', achieved_heights.shape)
            print('target-H', target_heights.shape)
            #Compute score according to Mean Cumulative Absolute Error 
            #if(metric == 'MCAE'):
            mcae_scores[b_i] = mean_cumulative_absolute_error(achieved_heights, target_heights, weights=None)
            if(args.display):
                print('MCAE :',mcae_scores[b_i])

            #elif(metric == 'MCSE'):
            mcse_scores[b_i] = mean_cumulative_squared_error(achieved_heights,target_heights,weights=None)
            if(args.display):
                print('MCSE :',mcse_scores[b_i])
            

        act_std_dev[:] = np.std(actions[:,1:],axis=0)
        # Map dead band
        actions[:,:] = np.where(actions[:,:] < 0., actions[:,:] - dead_band, actions[:,:] + dead_band)#+dead_band*np.sign(actions[b_i,:]))
        #actions[b_i,:] = np.where(np.abs(actions[b_i,:]) < (550.-dead_band), 0., actions[b_i,:])#+dead_band*np.sign(actions[b_i,:]))
        #actions[b_i,:] = np.where(np.abs(actions[b_i,:]) > (900.-dead_band), 900.-dead_band, actions[b_i,:])

        if plot_mult_tst_params and b_i==n_bench_itr-1:
            # Plot mean and variance of states over runs
            plt_i = 0
            ax = plt.subplot(sps[2*p_i + plt_i, 0])
            ax.title.set_text('Averaged over $%d$ itr, on %s, MCAE: $%.4f\pm%.3f$, \n MCSE $%.4f\pm%.3f$, Rew: $%.2f\pm%.3f$: \n'\
                '$\Delta u:%.3f\pm%.2fmA$, $\sigma(u_t): %.3f\pm%.3fmA$'%(
                n_bench_itr, tst_param, np.mean(mcae_scores), np.std(mcae_scores),
                np.mean(mcse_scores), np.std(mcse_scores),
                np.mean(cum_rews), np.std(cum_rews),
                np.mean(change_in_actions), np.std(change_in_actions),
                np.mean(act_std_dev), np.std(act_std_dev)
                ))

            #ax.title.set_text('averaged runs:')
            ax.plot(time_steps, np.mean(heights,axis=0), label="$\mu(x_t)$")
            ax.fill_between(time_steps, np.mean(heights,axis=0) - np.std(heights,axis=0),
                 np.mean(heights,axis=0) + np.std(heights,axis=0), color='blue', alpha=0.2, label="$\pm \sigma(x_t)$")
            ax.plot(time_steps, goals[b_i,:], label="$x_{des}$")
            ax.legend()
            ax.set_ylabel('$x_t$ in $m$')
            ax.set_xlabel('$t$ in steps of %s$s$'%(str(timeout)))

            plt_i += 1
            #ax = brokenaxes(ylims=((-750, -550),(550, 750)), subplot_spec=sps[2*p_i + plt_i, 0])
            ax = plt.subplot(sps[2*p_i + plt_i, 0])
            ax.plot(time_steps[1:], np.mean(actions[:,1:],axis=0), label="$\mu(u_t)$")
            ax.fill_between(time_steps[1:], np.mean(actions[:,1:],axis=0) - np.std(actions[:,1:],axis=0),
                 np.mean(actions[:,1:],axis=0) + np.std(actions[:,1:],axis=0), color='blue', alpha=0.2, label="$\pm\sigma(u_t)$")
            ax.set_ylabel('$u_t$ in $mA$')
            ax.set_xlabel('$t$ in steps of %s$s$'%(str(timeout)))
            ax.legend()
            plt_i += 1
            ax = plt.subplot(sps[2*p_i + plt_i, 0])
            ax.plot(time_steps, np.mean(rewards,axis=0), label="$\mu(R)$")
            ax.fill_between(time_steps, np.mean(rewards,axis=0) - np.std(rewards,axis=0),
                 np.mean(rewards,axis=0) + np.std(rewards,axis=0), color='blue', alpha=0.2, label="$\pm\sigma(R)$")
            ax.set_ylabel('$R$ ')
            ax.set_xlabel('$t$ in steps of %s$s$'%(str(timeout)))
            ax.legend()

            # Plot tracking results and compute score for chosen metric
            plt_i += 1
            ax = plt.subplot(sps[2*p_i + plt_i, 0])
            ax.title.set_text('sample run:')                                                
            ax.plot(time_steps, goals[b_i,:], label="$x_{des}$")
            ax.plot(time_steps, heights[b_i,:], label="x_t")
            ax.legend()
            ax.set_ylabel('$x_t$ in $m$')
            ax.set_xlabel('$t$ in steps of %s$s$'%(str(timeout)))
            plt_i += 1
            ax = plt.subplot(sps[2*p_i + plt_i, 0])
            #ax.plot(time_steps[1:], actions[b_i,1:], '-', label="$u_t$")
            ax.plot(time_steps[:], 550*np.ones(args.max_path_length), '--', color="black", label="$u_{db, env}$")
            ax.plot(time_steps[:], -550*np.ones(args.max_path_length), '--', color="black")
            ax.set_ylabel('$u_t$ in $mA$')
            ax.set_xlabel('$t$ in steps of %s$s$'%(str(timeout)))
            plt_i += 1
            ax = plt.subplot(sps[2*p_i + plt_i, 0])
            ax.plot(time_steps, rewards[b_i,:], '-', label="$R_t$")
            ax.set_ylabel('$R$ ')
            ax.set_xlabel('$t$ in steps of %s$s$'%(str(timeout)))

            # Execute and log deterministic rollout
            path = rollout(env, policy, max_path_length=args.max_path_length,
               animated=False, speedup=args.speedup, deterministic=True)
            state = {key:path['observations'][:,state_keys.index(key)] for key in state_keys}
            heights[b_i,:len(state['Height'])] = state['Height']
            actions[b_i,:len(path['actions'][:,0])] = path['actions'][:,0]
            rewards[b_i,:len(path['rewards'])] = path['rewards']
            actions[b_i,:] = lb + (actions[b_i,:] + 1.) * 0.5 * (ub - lb)
            actions[b_i,:] = np.clip(actions[b_i,:], lb, ub)
            change_in_actions_det = np.abs(actions[b_i,1:] - actions[b_i,:-1])
            actions[b_i,:] = np.clip(actions[b_i,:], lb, ub)
            actions[:,:] = np.where(actions[:,:] < 0., actions[:,:] - dead_band, actions[:,:] + dead_band)#+dead_band*np.sign(actions[b_i,:]))

            # Plot tracking results and compute score for chosen metric
            plt_i -= 2
            ax = plt.subplot(sps[2*p_i + plt_i, 0])
            ax.plot(time_steps, heights[b_i,:], '-',color='green', label="$x_{t, det}$")
            ax.legend()
            plt_i += 1
            ax = plt.subplot(sps[2*p_i + plt_i, 0])
            #ax.title.set_text('$\mu_{steps}(\Delta u):%.3f\pm%.3fmA$, $\sigma_\{steps\}(u_t): %.3fmA$'%(
            #    np.mean(change_in_actions_det), np.std(change_in_actions_det),
            #    np.std(actions[b_i,:])
            #    ))            
            ax.plot(time_steps[1:], actions[b_i,1:], '-',color='green',  label="$u_{t, det}$")
            ax.legend()
            plt_i += 1
            ax = plt.subplot(sps[2*p_i + plt_i, 0])
            ax.title.set_text('Rew: $%.2f$'%(np.sum(rewards[b_i,:])))
            ax.plot(time_steps, rewards[b_i,:], '-', color='green', label="$R_{t, det}$")
            ax.legend()


        elif b_i==n_bench_itr-1:
            print('not plotting mult test param')
            fig, ax = plt.subplots(nrows=2, ncols=1)
            
            ax[0].plot(time_steps, goals[b_i,:], label="target heights")
            ax[0].plot(time_steps, heights[b_i,:], label="reached heights")
            
            ax[1].plot(time_steps[1:], actions[b_i,1:], label="actions applied")
            plt.title(str(tst_param) + " Tracking Result")
            
        #fig = plt.figure(figsize=(5,5))
        #sps1, sps2 = GridSpec(2,1)
        #bax = brokenaxes(ylims=((-750, -550),(550, 750)), subplot_spec=sps1)
        #x = np.linspace(0, 1, 100)
        #bax.plot(time_steps[1:], np.mean(actions[:,1:],axis=0), label="$\mu(u_t)$")
        #bax.fill_between(time_steps[1:], np.mean(actions[:,1:],axis=0) - np.std(actions[:,1:],axis=0),
        #             np.mean(actions[:,1:],axis=0) + np.std(actions[:,1:],axis=0), color='blue', alpha=0.2, label="$\pm\sigma(u_t)$")

        #fig.tight_layout()
        plt.legend()

        total_steps = (train_itr_res)*eps_length*10 if eps_length else None
        
        plt_title = args.plt_title + ' ' + algo_name + '\n'\
        'NN ' + str([net_in_dim, net_shape, net_out_dim]) + '\n' \
        'train R:' + str(r_train) + ' on '+task+',\n'\
        '' + str(total_steps)+' stps:(' + str(train_itr_res) + 'eps*10runs*'+str(eps_length)+'stps)\n' \
        ''+rew_fn+' with $\\beta=$' + str(beta) + ', $\sigma=$' + str(sigma) +'\n'\
        'step_size='+str(step_size)+', test: '+'+'.join(args.tst_scenario)

        plt.suptitle(plt_title) # raised title
        if args.not_save_fig == False: 
            directory = "plots/" + args.model
            if args.tst_scenario: directory = directory + "_" + '_'.join(args.tst_scenario)
            if not os.path.exists(directory):
                os.makedirs(directory)
            plt.savefig(directory  + '.png')
        if(args.display):
            matplotlib.use( 'tkagg' )
            plt.show() 

        #if not query_yes_no('Continue simulation?'):
        #    break
