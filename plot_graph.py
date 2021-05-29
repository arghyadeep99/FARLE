import argparse
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.style as style
import os
import numpy as np
import re
import yaml

style.use('seaborn-poster')

#plot_mode = ['Individual', 'Comparison']
learning_mode = ['Scratch', 'Transfer']

parser = argparse.ArgumentParser()
parser.add_argument('--exp_prefix', type=str, default='untitled',
                    help='Enter your experiment\'s prefix name')
parser.add_argument('--learning_mode', type=str, default=None,
                    help=f'Enter learning mode: {",".join(learning_mode)}')
#parser.add_argument('--plotting_mode', type=str, default=None, help=f'Enter graph plotting mode: {",".join(plot_mode)}')
parser.add_argument('--graph_title', type=str, default=None,
                    help='Title to be used at top of graph')
parser.add_argument('--png_title', type=str, default=None,
                    help='Filename to save graph')

args = parser.parse_args()

plots_dir = 'plots'
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)


def get_total_hours(stringHMS):
    # stringHMS = "1d 0h 40m 20s"
    if int(stringHMS.split('h')[0].strip()) > 23:
        stringHMS = str(int(int(stringHMS.split('h')[0].strip()) / 24)) + "d " +str(int(stringHMS.split('h')[0].strip()) - 24) +"h "+stringHMS.split('h')[1][1:]
        timedeltaObj = dt.datetime.strptime(
            stringHMS, "%dd %Hh %Mm %Ss") - dt.datetime(1900,1, 1) + dt.timedelta(days=int(stringHMS.split('d')[0]))
    else :
        timedeltaObj = dt.datetime.strptime(
            stringHMS, "%Hh %Mm %Ss") - dt.datetime(1900, 1, 1)

    # print(timedeltaObj)
    return (timedeltaObj.total_seconds())/3600


def parse_logs(file):
    with open(file, 'r') as f:
        logs = f.read()
    episodes = []
    epsilon = []
    network_updates = []
    mean_rewards = []
    mean_lengths = []
    times = []
    for line in logs.split('\n')[:-1]:
        datum = line.split(', ')
        time = get_total_hours(datum[0].split(':')[1].strip())
        episode = int(datum[1].split(':')[1])
        eps = float(datum[2].split(':')[1])
        nu = int(datum[3].split(':')[1])
        emr = float(datum[4].split(':')[1])
        eml = float(datum[5].split(':')[1])
        episodes.append(episode)
        epsilon.append(eps)
        network_updates.append(nu)
        mean_rewards.append(emr)
        mean_lengths.append(eml)
        times.append(time)

    return episodes, epsilon, network_updates, mean_rewards, mean_lengths, times


def plot_graph(episodes, epsilon, network_updates, mean_rewards, mean_lengths, times):
    fig, ax = plt.subplots(2, 2, figsize=(16, 16))
    ax[0][0].plot(episodes, mean_rewards, color='blue', lw=1)
    ax[0][0].set_title('Mean Rewards per 100 episodes', fontsize=16)
    ax[0][0].set_xlabel('Number of episodes', fontsize=12)
    ax[0][0].set_ylabel('mean_rewards', fontsize=12)

    ax[0][1].plot(episodes, mean_lengths, color='green', lw=1)
    ax[0][1].set_title('Mean length per 100 episodes', fontsize=16)
    ax[0][1].set_xlabel('Number of episodes', fontsize=12)
    ax[0][1].set_ylabel('mean_length', fontsize=12)

    ax[1][0].plot(episodes, network_updates, color='red', lw=1)
    ax[1][0].set_title('Number of network updates', fontsize=16)
    ax[1][0].set_xlabel('Number of episodes', fontsize=12)
    ax[1][0].set_ylabel('network_updates', fontsize=12)

    ax[1][1].plot(episodes, times, color='violet', lw=1)
    ax[1][1].set_title('Time to train per 100 episodes', fontsize=16)
    ax[1][1].set_xlabel('Number of episodes', fontsize=12)
    ax[1][1].set_ylabel('time_to_train', fontsize=12)

    # plt.tight_layout()
    plt.suptitle(args.graph_title, fontsize=20)
    fig.savefig(f'./plots/{args.png_title}.png')
    # plt.show()


log_folders = os.listdir(os.path.join(os.getcwd(), 'logs'))
filtered_logs = []
if args.learning_mode == 'Scratch':
    for folder in log_folders:
        if (args.exp_prefix) in folder and ('Scratch' in folder or 'Resume' in folder) and ('Transfer' not in folder):
            filtered_logs.append(folder)
elif args.learning_mode == 'Transfer':
    for folder in log_folders:
        if ('Transfer' in folder or 'Resume' in folder) and (f'{args.exp_prefix}-from-' in folder):
            filtered_logs.append(folder)

filtered_logs.sort()
filtered_logs.insert(0, filtered_logs.pop(-1))
print(filtered_logs)
log_files_list = [os.path.join(
    os.getcwd(), 'logs', x, 'train.log') for x in filtered_logs]
print(log_files_list)


def combine_logs(log_files_list):
    mega_filename = f'./plots/{args.exp_prefix}-{args.learning_mode}-combined_train.log'
    lastTime = ""
    with open(mega_filename, 'w') as f, open(log_files_list[0], 'r') as f1:
        scratch_log = f1.readlines()
        # f.write(scratch_log)
        mean_rewards = []
        for line in scratch_log:
            datum = line.split(', ')
            emr = float(datum[4].split(':')[1])
            mean_rewards.append(emr)

        max_reward_index = mean_rewards.index(max(mean_rewards))
        lastTime = scratch_log[max_reward_index -
                               1].split(',')[0].split(':')[1].strip()
        lastH = int(lastTime.split('h')[0])
        lastM = int(lastTime.split('h')[1].split('m')[0])
        lastS = int(lastTime.split('h')[1].split('m')[1].split('s')[0])
        # print(lastH,lastM,lastS)
        f.write(''.join(scratch_log[:max_reward_index]))

    mega_file = open(mega_filename, 'a')
    for log_file in log_files_list[1:]:
        with open(log_file, 'r') as f:
            resume_log = f.readlines()
            mean_rewards = []
            updated_time_log = []
            for line in resume_log:
                datum = line.split(', ')
                emr = float(datum[4].split(':')[1])
                mean_rewards.append(emr)
                newTime = line.split(',')[0].split(':')[1].strip()
                newH = int(newTime.split('h')[0])
                newM = int(newTime.split('h')[1].split('m')[0])
                newS = int(newTime.split('h')[1].split('m')[1].split('s')[0])
                # print(newH,newM,newS)
                newH, newM, newS = lastH+newH, lastM+newM, lastS+newS
                if newS > 60:
                    newM = int(newM + newS/60)
                    newS = newS % 60
                if newM > 60:
                    newH = int(newH + newM/60)
                    newM = newM % 60
                newLine = "time elapsed: " + \
                    str(newH)+"h "+str(newM)+"m " + \
                    str(newS)+"s,"+line.split(',', 1)[1]
                updated_time_log.append(newLine)

            print(max(mean_rewards))
            max_reward_index = mean_rewards.index(max(mean_rewards))
            mega_file.write(''.join(updated_time_log[:max_reward_index]))
            lastTime = updated_time_log[max_reward_index-1].split(',')[0].split(':')[1].strip()
            lastH = int(lastTime.split('h')[0])
            lastM = int(lastTime.split('h')[1].split('m')[0])
            lastS = int(lastTime.split('h')[1].split('m')[1].split('s')[0])
            # print("change")
            # print(lastH,lastM,lastS)
        
    mega_file.close()
    return mega_filename


mega_filename = combine_logs(log_files_list)
episodes, epsilon, network_updates, mean_rewards, mean_lengths, times = parse_logs(
    mega_filename)
plot_graph(episodes, epsilon, network_updates,
           mean_rewards, mean_lengths, times)
