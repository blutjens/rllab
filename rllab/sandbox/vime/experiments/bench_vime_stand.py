import argparse

import joblib
import tensorflow as tf

from rllab.misc.console import query_yes_no
from rllab.sampler.utils import rollout

import os
os.environ["THEANO_FLAGS"] = "device=cpu"
import numpy as np
import matplotlib.pyplot as plt

from solenoid.envs.constants import state_keys
from solenoid.envs import constants

from solenoid.misc.metrics import mean_cumulative_absolute_error, mean_cumulative_squared_error
from solenoid.misc.tasks import SineTask,ChirpTask,StepTask#,ToZeroTask

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='data/logs_vime_stand/itr_54.pkl',
                        help='path to the snapshot file')
    parser.add_argument('--max_path_length', type=int, default=500,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=1,
                        help='Speedup')
    parser.add_argument('--display', type=bool, default=True,
                        help='plot and print results')
    args = parser.parse_args()

    # If the snapshot file use tensorflow, do:
    # import tensorflow as tf
    # with tf.Session():
    #     [rest of the code]
    with tf.Session() as sess:
        data = joblib.load(args.file)
        policy = data['policy']
        env = data['env']

        tst_params = ['sine']#, 'chirp', 'step']
        plot_mult_tst_params = True
        plt.rcParams.update({'font.size': 28})
        fig, ax = plt.subplots(nrows=len(tst_params)*2, ncols=1,figsize=(len(tst_params)*10,len(tst_params)*10*2), dpi=80 )

        for p_i, tst_param in enumerate(tst_params):

            env.wrapped_env.sigma = 0.05
            print('env.w', env.wrapped_env.sigma)
            # Execute and log 
            path = rollout(env, policy, max_path_length=args.max_path_length,
                           animated=False, speedup=args.speedup)

            # 
            print('path keys', path.keys())
            state = {key:path['observations'][:,state_keys.index(key)] for key in state_keys}
            actions = path['actions']
            rewards = path['rewards']
            print('rewards', rewards)
            print('rew', np.sum(np.array(rewards)))
            # Get env params
            time_steps = np.arange(state['Height'].shape[0])
            print('ts', time_steps)
            timeout = env.wrapped_env.timestep
            # Plot tracking results and compute score for chosen metric
            if plot_mult_tst_params:
                ax[2*p_i + 0].plot(time_steps, state['Goal_Height'], label="target heights")
                ax[2*p_i + 0].plot(time_steps, state['Height'], label="reached heights")
                ax[2*p_i + 0].set_ylabel('$x_t$ in $m$')
                ax[2*p_i + 0].set_xlabel('$t$ in steps of %s$s$'%(str(timeout)))
                ax[2*p_i + 1].plot(time_steps, actions, '-', label="actions applied")
                ax[2*p_i + 1].set_ylabel('$u_t$ in $mA$')
                ax[2*p_i + 1].set_xlabel('$t$ in steps of %s$s$'%(str(timeout)))
                ax[2*p_i + 2].plot(time_steps, rewards, '-', label="actions applied")
                ax[2*p_i + 2].set_ylabel('$R$ ')
                ax[2*p_i + 2].set_xlabel('$t$ in steps of %s$s$'%(str(timeout)))

            else:
                fig, ax = plt.subplots(nrows=2, ncols=1)
                
                ax[0].plot(time_steps, state['Goal_Height'], label="target heights")
                ax[0].plot(time_steps, state['Height'], label="reached heights")
                
                ax[1].plot(time_steps, actions, label="actions applied")
                plt.title(str(tst_param) + " Tracking Result")

            #Convert to shape [episodes *time * number of height dimensions] for metric computation
            achieved_heights = np.expand_dims(np.expand_dims(np.array(state['Height']).T,axis=0),axis=2)
            target_heights = np.expand_dims(np.expand_dims(np.array(state['Goal_Height']).T, axis = 0),axis=2)
            print('ach h', achieved_heights.shape)
            print('target-H', target_heights.shape)
            #Compute score according to Mean Cumulative Absolute Error 
            #if(metric == 'MCAE'):
            mcae_score = mean_cumulative_absolute_error(achieved_heights, target_heights, weights=None)
            if(args.display):
                print('MCAE :',mcae_score)

            #elif(metric == 'MCSE'):
            mcse_score = mean_cumulative_squared_error(achieved_heights,target_heights,weights=None)
            if(args.display):
                print('MCSE :',mcse_score)

            ax[2*p_i + 0].title.set_text('LQT on %s response, MCAE: %.4f, MCSE %.4f'%(
                tst_param, mcae_score, mcse_score))

            plt.suptitle("trpo+vime")
            #plt.savefig(control + str(round(time.time())) + ".png")
            if(args.display):
                #plt.show() 
                plt.savefig('plots/tst.png')

            #if not query_yes_no('Continue simulation?'):
            #    break
