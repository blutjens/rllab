import argparse

import joblib
import tensorflow as tf

from rllab.misc.console import query_yes_no
from rllab.sampler.utils import rollout

import os
os.environ["THEANO_FLAGS"] = "device=cpu"
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

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
    if args.display:
        print('env', env)
        print('settin gvis true')
        env.wrapped_env.vis = True
    else:
        print('not disp')
        env.wrapped_env.vis = False
    n_bench_itr = 20

    tst_params = ['sine']#, 'chirp', 'step']
    plot_mult_tst_params = True
    plt.rcParams.update({'font.size': 28})
    
    n_plts_p_param = 6
    fig, ax = plt.subplots(nrows=len(tst_params)*n_plts_p_param, ncols=1,figsize=(20,len(tst_params)*10*n_plts_p_param), dpi=80 )

    for p_i, tst_param in enumerate(tst_params):
        #import matplotlib.gridspec as gridspec
        #outer = gridspec.GridSpec(len(tst_params)*n_plts_p_param, 1) 
        #make nested gridspecs
        #gs1 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec = outer[0])
        #gs2 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec = outer[1], hspace = .05)

        # Get env/net/policy params for printing
        algo_name_long = data['algo'].__class__.__module__ + '.' + data['algo'].__class__.__name__
        vime = '+ VIME' if 'vime' in algo_name_long else ''
        trpo = 'TRPO' if 'trpo' in algo_name_long else ''
        algo_name = trpo + vime
        print('data', data, vime)
        if vime != '':
            r_train = np.mean(data['episode_rewards'])
            eps_length = int(np.mean(data['episode_lengths']))
        else:  
            r_train = None
            eps_length = None
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
        env.wrapped_env.task = SineTask(
                steps=500, periods=2., offset=0.)

        mcae_scores = np.zeros(n_bench_itr)
        cum_rews = np.zeros(n_bench_itr)
        mcse_scores = np.zeros(n_bench_itr)
        heights = np.zeros((n_bench_itr, args.max_path_length))
        goals = np.zeros((n_bench_itr, args.max_path_length))
        actions = np.zeros((n_bench_itr, args.max_path_length))
        rewards = np.zeros((n_bench_itr, args.max_path_length))
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
            time_steps = np.arange(state['Height'].shape[0])
            # Get env params
            print('ts', time_steps.shape)
            print('rew', rewards.shape)
            timeout = env.wrapped_env.timestep

            # Scale actions to [-max_ma, max_ma]
            lb, ub = env.wrapped_env.action_space.bounds
            print('action bnd', lb, ub)
            print('actions', actions[b_i, :10])
            actions[b_i,:] = lb + (actions[b_i,:] + 1.) * 0.5 * (ub - lb)
            actions[b_i,:] = np.clip(actions[b_i,:], lb, ub)
            print('actions', actions[b_i, :10])
            # Map dead band
            #actions[b_i,:] = np.where(np.abs(actions[b_i,:]) < (550.-dead_band), 0., actions[b_i,:])#+dead_band*np.sign(actions[b_i,:]))
            #actions[b_i,:] = np.where(np.abs(actions[b_i,:]) > (900.-dead_band), 900.-dead_band, actions[b_i,:])

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

            if plot_mult_tst_params and b_i==n_bench_itr-1:
                # Plot mean and variance of states over runs
                plt_i = 0
                ax[2*p_i + plt_i].title.set_text('averaged runs:')

                ax[2*p_i + plt_i].plot(time_steps, np.mean(heights,axis=0), label="$\mu(x_t)$")
                ax[2*p_i + plt_i].fill_between(time_steps, np.mean(heights,axis=0) - np.std(heights,axis=0),
                     np.mean(heights,axis=0) + np.std(heights,axis=0), color='blue', alpha=0.2, label="$\pm \sigma(x_t)$")
                ax[2*p_i + plt_i].plot(time_steps, goals[b_i,:], label="$x_{des}$")
                ax[2*p_i + plt_i].legend()
                ax[2*p_i + plt_i].set_ylabel('$x_t$ in $m$')
                ax[2*p_i + plt_i].set_xlabel('$t$ in steps of %s$s$'%(str(timeout)))
                plt_i += 1
                ax[2*p_i + plt_i].plot(time_steps[1:], np.mean(actions[:,1:],axis=0), label="$\mu(u_t)$")
                ax[2*p_i + plt_i].fill_between(time_steps[1:], np.mean(actions[:,1:],axis=0) - np.std(actions[:,1:],axis=0),
                     np.mean(actions[:,1:],axis=0) + np.std(actions[:,1:],axis=0), color='blue', alpha=0.2, label="$\pm\sigma(u_t)$")
                ax[2*p_i + plt_i].set_ylabel('$u_t$ in $mA$')
                ax[2*p_i + plt_i].set_xlabel('$t$ in steps of %s$s$'%(str(timeout)))
                ax[2*p_i + plt_i].legend()
                plt_i += 1
                ax[2*p_i + plt_i].plot(time_steps, np.mean(rewards,axis=0), label="$\mu(R)$")
                ax[2*p_i + plt_i].fill_between(time_steps, np.mean(rewards,axis=0) - np.std(rewards,axis=0),
                     np.mean(rewards,axis=0) + np.std(rewards,axis=0), color='blue', alpha=0.2, label="$\pm\sigma(R)$")
                ax[2*p_i + plt_i].set_ylabel('$R$ ')
                ax[2*p_i + plt_i].set_xlabel('$t$ in steps of %s$s$'%(str(timeout)))
                ax[2*p_i + plt_i].legend()
                            
                ax[2*p_i + 0].title.set_text('Averaged over $%d$ itr, on %s, MCAE: $%.4f\pm%.3f$, \n MCSE $%.4f\pm%.3f$, Rew: $%.2f\pm%.3f$: \n'%(
                n_bench_itr, tst_param, np.mean(mcae_scores), np.std(mcae_scores),
                np.mean(mcse_scores), np.std(mcse_scores),
                np.mean(cum_rews), np.std(cum_rews)))

                # Plot tracking results and compute score for chosen metric
                plt_i += 1
                ax[2*p_i + plt_i].title.set_text('sample run:')                                                
                ax[2*p_i + plt_i].plot(time_steps, goals[b_i,:], label="$x_{des}$")
                ax[2*p_i + plt_i].plot(time_steps, heights[b_i,:], label="x_t")
                ax[2*p_i + plt_i].legend()
                ax[2*p_i + plt_i].set_ylabel('$x_t$ in $m$')
                ax[2*p_i + plt_i].set_xlabel('$t$ in steps of %s$s$'%(str(timeout)))
                plt_i += 1
                ax[2*p_i + plt_i].plot(time_steps[1:], actions[b_i,1:], '-', label="actions applied")
                ax[2*p_i + plt_i].set_ylabel('$u_t$ in $mA$')
                ax[2*p_i + plt_i].set_xlabel('$t$ in steps of %s$s$'%(str(timeout)))
                plt_i += 1
                ax[2*p_i + plt_i].plot(time_steps, rewards[b_i,:], '-', label="actions applied")
                ax[2*p_i + plt_i].set_ylabel('$R$ ')
                ax[2*p_i + plt_i].set_xlabel('$t$ in steps of %s$s$'%(str(timeout)))
            elif b_i==n_bench_itr-1:
                print('not plotting mult test param')
                fig, ax = plt.subplots(nrows=2, ncols=1)
                
                ax[0].plot(time_steps, goals[b_i,:], label="target heights")
                ax[0].plot(time_steps, heights[b_i,:], label="reached heights")
                
                ax[1].plot(time_steps[1:], actions[b_i,1:], label="actions applied")
                plt.title(str(tst_param) + " Tracking Result")

        #fig.tight_layout()
        plt.legend()

        total_steps = (train_itr_res+44)*eps_length*10 if eps_length else None
        

        plt_title = args.plt_title + ' ' + algo_name + '\n'\
        'NN ' + str([net_in_dim, net_shape, net_out_dim]) + '\n' \
        'train R:' + str(r_train) + ' on '+task+',\n'\
        '' + str(total_steps)+' stps:(' + str(train_itr_res+44) + 'eps*10runs*'+str(eps_length)+'stps)\n' \
        ''+rew_fn+' with $\\beta=$' + str(beta) + ', $\sigma=$' + str(sigma)


        plt.suptitle(plt_title) # raised title

        directory = "plots/" + args.model
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(directory  + '.png')
        if(args.display):
            matplotlib.use( 'tkagg' )
            plt.show() 

        #if not query_yes_no('Continue simulation?'):
        #    break
