import argparse

import joblib
import tensorflow as tf
from tqdm import tqdm 

from rllab.misc.console import query_yes_no
from rllab.sampler.utils import rollout

import os
os.environ["THEANO_FLAGS"] = "device=cpu"
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
        
from solenoid.envs.constants import state_keys
from solenoid.envs import constants
from solenoid.misc.metrics import mean_cumulative_absolute_error, mean_cumulative_squared_error
from solenoid.misc.tasks import SineTask,ChirpTask,StepTask#,ToZeroTask

if __name__ == "__main__":
    """
    This file creates plots to evaluate already trained policies
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_folder', type=str, default='data',
                        help='path to the snapshot folder')
    parser.add_argument('--model', type=str, default='logs_vime_stand_4_0_res/itr_17',
                        help='path to the snapshot file')
    parser.add_argument('--plt_title', type=str, default='',
                        help='plot title')
    parser.add_argument('--max_path_length', type=int, default=500,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=1,
                        help='Speedup')
    parser.add_argument('--display', action="store_true", default=False,
                        help='Enable visualization of environment and show plot (requires GUI forwarding from docker)')
    parser.add_argument('--not_save_fig', action="store_false", default=False,
                        help='save plot in file')
    parser.add_argument('--deterministic', action="store_true", default=False,
                        help='average over multiple deterministic rollouts')
    parser.add_argument('--n_eval_eps', type=int, default=1,
                        help='number of evaluation episodes')
    parser.add_argument('--plt_u_over_train_itr', action="store_true", default=False,
                        help='create plot over multiple training iterations')
    parser.add_argument('--not_verbose', action="store_true", default=False,
                        help='not print logs of environment to terminal')
    parser.add_argument('--bench_maml', action="store_true", default=False,
                        help='benchmark maml results')
    parser.add_argument('--tst_scenario', action="store", nargs='*', type=str, default=None,
                        choices=['train', 'sine', 'chirp', 'step', 'larger_deadbands', 'no_deadbands', 
                        'lower_height_rate_up', 'greater_height_rate_up', 'learn_lqt_plus_rl', 'real', 
                        'just_lqt', 'rew_action_penalty', 'rew_goal_bias'],
                        help='test scenario')
    args = parser.parse_args()

    # Initialize gif writer, if gif over multiple training iterations is to be created.
    if args.plt_u_over_train_itr:
        import imageio
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        itrs_to_plot =  np.linspace(0, 225, 112, dtype=int)#np.linspace(299)#[1, 49, 99, 149, 199]
        model_folder = args.model_folder + "/" + args.model
        print(str(model_folder))
        model_files = ['' + model_folder + '/itr_' + str(itr) + '.pkl' for itr in itrs_to_plot]
        directory = 'plots/'+ args.model
        if not os.path.exists(directory):
            os.makedirs(directory)
        gif_writer = imageio.get_writer(directory +'/u_over_train_itr.gif', mode='I')
        args.not_save_fig = True
        args.display = False
    else:
        model_files = [args.model_folder + '/' + args.model + '.pkl']

    for model_file in tqdm(model_files):
      print('model_file', model_file)
      if args.bench_maml: import tensorflow as tf
      # TODO uncomment this line, if benchmarking MAML
      #with tf.Session() as sess:
    data = joblib.load(model_file)
    policy = data['policy']
    env = data['env']
    # Init env action space (TODO find out when this is necessary)
    #env.action_space
    env.log_tb = False # Don't create tensorboard logs during benchmark
    if args.display:
        env.wrapped_env.vis = True
    else:
        env.wrapped_env.vis = False
    n_eval_eps = args.n_eval_eps
    tst_params = ['sine']#, 'chirp', 'step'] 

    # Initialize plot canvas
    plot_mult_tst_params = True
    plt.rcParams.update({'font.size': 28})
    n_plts_p_param = 6 if not args.plt_u_over_train_itr else 3
    fig_height = len(tst_params)*10*n_plts_p_param 
    fig = plt.figure(figsize=(20,fig_height), dpi=80 )
    sps = GridSpec(nrows=len(tst_params)*n_plts_p_param, ncols=1)
    if args.plt_u_over_train_itr: 
        canvas = FigureCanvas(fig)
        plt.subplots_adjust(top=0.8)

    # Read in all environment, and policy parameters for plotting
    for p_i, tst_param in enumerate(tst_params):
        print('algo', data)
        if args.bench_maml:
            algo_name = 'maml'
            step_size = 0.005
            eps_length = 500
            batch_size = 5000
            env = env.wrapped_env
            net_in_dim = 11
            net_out_dim = 1
            net_shape = (64, 32)
        else:
            algo_name_long = data['algo'].__class__.__module__ + '.' + data['algo'].__class__.__name__
            trpo = 'TRPO ' if 'trpo' in algo_name_long else ''
            vime = '+ VIME ' if 'vime' in algo_name_long else ''
            lqt = '+ LQT ' if env.wrapped_env.learn_lqt_plus_rl else ''
            algo_name = trpo + vime + lqt
            step_size = data['algo'].step_size
            #if vime != '':
            #    r_train = None#np.mean(data['episode_rewards'])
            #    eps_length = int(np.mean(data['episode_lengths']))
            #else:  
            eps_length = int(data['algo'].max_path_length)
            batch_size = int(data['algo'].batch_size)
            net_in_dim = policy._mean_network._layers[0].shape[1]
            net_out_dim = policy._mean_network._layers[-1].num_units
            net_shape = [layer.num_units for layer in policy._mean_network._layers[1:-1]]

        avg_eps_p_itr = int(batch_size / eps_length)
        r_train = None
        train_itr_res = data['itr']
        task = env.wrapped_env.task.__class__.__name__
        beta = env.wrapped_env.beta
        sigma = env.wrapped_env.sigma
        rew_fn = env.wrapped_env.reward_fn.__name__
        dead_band = env.wrapped_env.dead_band 
        env.wrapped_env.sigma = 0.05 # Ensure sigma of goal_bias reward function is set 
        print('set eval reward fn sigma to ', env.wrapped_env.sigma)
        if args.not_verbose: env.wrapped_env.verbose = False
        # Test different random seeds:
        #seed = env.wrapped_env.seed
        #np.random.seed(seed)
        #random.seed(seed)


        # Read in arguments for test under different task, transition dynamics, environments, ...
        if "train" in args.tst_scenario:
            pass
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
            # TODO set this path with controls module path
            dynamics_path = '../../../../../solenoid/controls/data/teststand_AB_opt_on_sim.pkl'
            env.wrapped_env.lqt = LQT(dynamics_path)
        # Test under different reward function
        if "rew_action_penalty" in args.tst_scenario:
            from solenoid.misc.reward_fns import goal_bias_action_penalty_2
            env.wrapped_env.reward_fn = goal_bias_action_penalty_2
        if "rew_goal_bias" in args.tst_scenario:
            from solenoid.misc.reward_fns import goal_bias
            env.wrapped_env.reward_fn = goal_bias
        # Test on real teststand
        if "real" in args.tst_scenario:
            from rllab.sandbox.vime.envs.test_stand import TestStandReal
            env.wrapped_env.sim = "real"
            env.wrapped_env.test_stand = TestStandReal(env=env.wrapped_env, use_proxy=env.wrapped_env.use_proxy)
        if "just_lqt" in args.tst_scenario:
            just_lqt = True
        else:
            just_lqt = False
        env.wrapped_env.just_lqt = just_lqt
        if just_lqt: algo_name = 'LQT'            

        # Initialize variables for plotting
        mcae_scores = np.zeros(n_eval_eps)
        cum_rews = np.zeros(n_eval_eps)
        mcse_scores = np.zeros(n_eval_eps)
        heights = np.zeros((n_eval_eps, args.max_path_length))
        goals = np.zeros((n_eval_eps, args.max_path_length))
        actions = np.zeros((n_eval_eps, args.max_path_length))
        actions_rl = np.zeros((n_eval_eps, args.max_path_length))
        postprocessed_actions = np.zeros((n_eval_eps, args.max_path_length))
        if env.wrapped_env.learn_lqt_plus_rl: 
            actions_lqt = np.zeros((n_eval_eps, args.max_path_length)) 
        rewards = np.zeros((n_eval_eps, args.max_path_length))
        if env.wrapped_env.reward_fn.__name__ == 'goal_bias_action_penalty_2':
            action_penalties = np.zeros((n_eval_eps, args.max_path_length)) 
            goal_rews = np.zeros((n_eval_eps, args.max_path_length)) 
        change_in_actions = np.zeros((n_eval_eps))
        act_std_dev = np.zeros((args.max_path_length-1)) 

        # Average over multiple rollouts in stochastic environments
        for b_i in tqdm(range(n_eval_eps)):
            # Execute and log 
            path = rollout(env, policy, max_path_length=args.max_path_length,
                           animated=False, speedup=args.speedup, deterministic=args.deterministic)

            # Read states
            state = {key:path['observations'][:,state_keys.index(key)] for key in state_keys}
            heights[b_i,:len(state['Height'])] = state['Height']
            goals[b_i,:len(state['Goal_Height'])] = state['Goal_Height']
            actions[b_i,:len(path['actions'][:,0])] = path['actions'][:,0]
            rewards[b_i,:len(path['rewards'])] = path['rewards']
            cum_rews[b_i] = np.sum(np.array(rewards[b_i,:]))
            time_steps = np.arange(args.max_path_length)
            timeout = env.wrapped_env.timestep

            # Get Rewards, if reward is shaped:
            if env.wrapped_env.reward_fn.__name__ == 'goal_bias_action_penalty_2':
                action_penalties[b_i,:] = path['env_infos']['action_penalty'][:].reshape((path['env_infos']['action_penalty'].shape[0],))
                goal_rews[b_i,:] = path['env_infos']['goal_rew'][:].reshape((path['env_infos']['goal_rew'].shape[0],))

            # Scale actions to [-max_ma, max_ma], as calculated in rllab->rllab->envs->normalized_env.py->step()
            lb, ub = env.wrapped_env.action_space.bounds
            actions[b_i,:] = lb + (actions[b_i,:] + 1.) * 0.5 * (ub - lb)
            actions[b_i,:] = np.clip(actions[b_i,:], lb, ub)
            actions_rl[b_i,:] = actions[b_i,:]
            postp_act = path['env_infos']['taken_action'][:].reshape((path['env_infos']['taken_action'].shape[0],))
            postprocessed_actions[b_i,:len(postp_act)] = postp_act
            if env.wrapped_env.learn_lqt_plus_rl: 
                act_rl_plus_lqt_minus_db = postprocessed_actions[b_i,:] - np.sign(postprocessed_actions[b_i,:])*dead_band
                actions_lqt[b_i,:] = act_rl_plus_lqt_minus_db - actions[b_i,:]

            # Calculate change and std dev in actions
            change_in_actions[b_i] = np.mean(np.abs(actions[b_i,1:] - actions[b_i,:-1]))

            #Convert to shape [episodes *time * number of height dimensions] for metric computation
            achieved_heights = np.expand_dims(np.expand_dims(np.array(heights[b_i,:]).T,axis=0),axis=2)
            target_heights = np.expand_dims(np.expand_dims(np.array(goals[b_i,:]).T, axis = 0),axis=2)
            
            #Compute scores according to Mean Cumulative Absolute Errors (MCAE), or Mean Cumulative Squared Error (MCSE)
            mcae_scores[b_i] = mean_cumulative_absolute_error(achieved_heights, target_heights, weights=None)
            if(args.display): print('MCAE :',mcae_scores[b_i])
            mcse_scores[b_i] = mean_cumulative_squared_error(achieved_heights,target_heights,weights=None)
            if(args.display): print('MCSE :',mcse_scores[b_i])
            

        act_std_dev[:] = np.std(actions[:,1:],axis=0)
        # Map dead band
        actions[:,:] = np.where(actions[:,:] < 0., actions[:,:] - dead_band, actions[:,:] + dead_band)

        if plot_mult_tst_params and b_i==n_eval_eps-1:
            # Plot mean and variance of states over runs
            plt_i = 0
            ax = plt.subplot(sps[2*p_i + plt_i, 0])
            ax.title.set_text('Averaged over $%d$ itr, MCAE: %07.4f$\pm$%5.3f, \n MCSE $%08.4f\pm%.3f$, Rew: $%08.2f\pm%.3f$, Det.Pol.: %r\n'\
                '$\Delta u:%07.3f\pm%.2fmA$, $\sigma(u_t): %.3f\pm%.3fmA$'%(
                n_eval_eps, np.mean(mcae_scores), np.std(mcae_scores),
                np.mean(mcse_scores), np.std(mcse_scores),
                np.mean(cum_rews), np.std(cum_rews),
                args.deterministic,
                np.mean(change_in_actions), np.std(change_in_actions),
                np.mean(act_std_dev), np.std(act_std_dev)
                ))
            #ax.title.set_text('averaged runs:')
            ax.plot(time_steps, np.mean(heights,axis=0), label="$\mu(x_t)$")
            ax.fill_between(time_steps, np.mean(heights,axis=0) - np.std(heights,axis=0),
                 np.mean(heights,axis=0) + np.std(heights,axis=0), color='C0', alpha=0.2, label="$\pm \sigma(x_t)$")
            ax.plot(time_steps, goals[b_i,:], label="$x_{des}$")
            ax.legend(loc='upper right')
            ax.set_ylim((constants.height_min, constants.height_max))
            ax.set_ylabel('$x_t$ in $m$')
            ax.set_xlabel('$t$ in steps of %s$s$'%(str(timeout)))

            # Plot actions over time
            plt_i += 1
            ax = plt.subplot(sps[2*p_i + plt_i, 0])
            if not just_lqt: ax.plot(time_steps[1:], np.mean(actions_rl[:,1:],axis=0), color='green', label="$\mu(u_{t,rl})$")
            if env.wrapped_env.learn_lqt_plus_rl and not just_lqt: ax.plot(time_steps[1:], np.mean(actions_lqt[:,1:],axis=0), color='orange', label="$\mu(u_{t,lqt})$")
            ax.plot(time_steps[1:], np.mean(postprocessed_actions[:,1:],axis=0), color='C0', label="$\mu(u_t)$")
            ax.fill_between(time_steps[1:], np.mean(postprocessed_actions[:,1:],axis=0) - np.std(postprocessed_actions[:,1:],axis=0),
                 np.mean(postprocessed_actions[:,1:],axis=0) + np.std(postprocessed_actions[:,1:],axis=0), color='C0', alpha=0.2, label="$\pm\sigma(u_t)$")
            ax.set_ylabel('$u_t$ in $mA$')
            ax.set_xlabel('$t$ in steps of %s$s$'%(str(timeout)))
            ax.legend(loc='upper right')

            # Plot rewards over time 
            plt_i += 1
            ax = plt.subplot(sps[2*p_i + plt_i, 0])
            ax.plot(time_steps, np.mean(rewards,axis=0), label="$\mu(R)$")
            ax.fill_between(time_steps, np.mean(rewards,axis=0) - np.std(rewards,axis=0),
                 np.mean(rewards,axis=0) + np.std(rewards,axis=0), color='C0', alpha=0.2, label="$\pm\sigma(R)$")
            if env.wrapped_env.reward_fn.__name__ == 'goal_bias_action_penalty_2':
                ax.plot(time_steps, np.mean(action_penalties,axis=0), label="$\mu(R_u)$")
                ax.plot(time_steps, np.mean(goal_rews,axis=0), label="$\mu(R_{x_g})$")
            if args.plt_u_over_train_itr : ax.set_ylim((0.90, 1.))
            ax.set_ylabel('$R$ ')
            ax.set_xlabel('$t$ in steps of %s$s$'%(str(timeout)))
            ax.legend(loc='lower right')

            # Create three more plots to plot one sample run in a stochastic environment
            if not args.plt_u_over_train_itr:
                # Plot tracking results and compute score for chosen metric
                plt_i += 1
                ax = plt.subplot(sps[2*p_i + plt_i, 0])
                ax.title.set_text('sample run:')
                ax.plot(time_steps, goals[b_i,:], color='orange', label="$x_{des}$")
                ax.plot(0, 0.55, '*', label="$x_{init}$")
                if args.deterministic: ax.plot(time_steps, heights[b_i,:], color='C0', label="x_t")
                ax.legend()
                ax.set_ylabel('$x_t$ in $m$')
                ax.set_xlabel('$t$ in steps of %s$s$'%(str(timeout)))
                plt_i += 1
                ax = plt.subplot(sps[2*p_i + plt_i, 0])
                #ax.plot(time_steps[1:], actions[b_i,1:], '-', label="$u_t$")
                if args.deterministic:
                    if not just_lqt: ax.plot(time_steps[1:], actions_rl[b_i,1:], '-',color='green',  label="$u_{t, rl}$")
                    if env.wrapped_env.learn_lqt_plus_rl and not just_lqt: ax.plot(time_steps[1:], actions_lqt[b_i,1:], '-',color='orange',  label="$u_{t, lqt}$")
                    ax.plot(time_steps[1:], postprocessed_actions[b_i,1:], '-',color='C0',  label="$u_{t}$")
                ax.plot(time_steps[:], 550*np.ones(args.max_path_length), '--', color='black', label="$u_{db, env}$")
                ax.plot(time_steps[:], -550*np.ones(args.max_path_length), '--', color='black')
                ax.set_ylabel('$u_t$ in $mA$')
                ax.set_xlabel('$t$ in steps of %s$s$'%(str(timeout)))
                ax.legend()
                plt_i += 1
                ax = plt.subplot(sps[2*p_i + plt_i, 0])
                if args.deterministic: 
                    ax.title.set_text('Rew: $%.2f$'%(np.sum(rewards[b_i,:])))
                    ax.plot(time_steps, rewards[b_i,:], '-', label="$R_t$")
                ax.set_ylabel('$R$ ')
                ax.set_xlabel('$t$ in steps of %s$s$'%(str(timeout)))
                ax.legend()

                # Execute and log deterministic rollout (this is only necessary when the environment, or policy at test time is stochastic)
                if not args.deterministic:
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
                    actions[:,:] = np.where(actions[:,:] < 0., actions[:,:] - dead_band, actions[:,:] + dead_band)

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


        elif b_i==n_eval_eps-1:
            print('not plotting mult test param')
            fig, ax = plt.subplots(nrows=2, ncols=1)
            
            ax[0].plot(time_steps, goals[b_i,:], label="target heights")
            ax[0].plot(time_steps, heights[b_i,:], label="reached heights")
            
            ax[1].plot(time_steps[1:], actions[b_i,1:], label="actions applied")
            plt.title(str(tst_param) + " Tracking Result")
            
        total_steps = (train_itr_res)*eps_length*avg_eps_p_itr if eps_length else None
        
        plt_title = args.plt_title + ' ' + algo_name + '\n'\
        'NN ' + str([net_in_dim, net_shape, net_out_dim]) + '\n' \
        'train R:' + str(r_train) + ' on '+task+',\n'\
        '%010d'%(total_steps)+' stps:(%04d'%(train_itr_res) + 'itr*'+ str(avg_eps_p_itr)+'eps*'+str(eps_length)+'stps)\n' \
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

        if args.plt_u_over_train_itr:
            plt.draw()
            canvas.draw()       # draw the canvas, cache the renderer
            width, height = (fig.get_size_inches() * fig.get_dpi()).astype(int)
            img = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3) 
            plt.imshow(img)
            gif_writer.append_data(img)
