import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib.lines import Line2D
import matplotlib


matplotlib.rcParams['text.usetex'] = True
matplotlib.rc('font',family='serif', serif=['Palatino'])
sns.set(font='serif', font_scale=1.4)
sns.set_style("white", {
        "font.family": "serif",
        "font.weight": "normal",
        "font.serif": ["Times", "Palatino", "serif"],
        'axes.facecolor': 'white',
        'lines.markeredgewidth': 1})
def setup_plot():
    fig = plt.figure(dpi=100, figsize=(5.0,3.0))
    ax = plt.subplot(111)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis = 'both', which = 'major', labelsize = 15)
    ax.tick_params(axis = 'both', which = 'minor', labelsize = 15)
    ax.tick_params(direction='in')


orange = "#F79646"
teal = "#4BACC6"
grey = "grey"
green = "#008000"
purple = "#8064A2"
x = (5 * np.arange(450)) * 100

ang_rand = np.load("hopper/angular_logs_random_1.npy")
ang_ours = np.load("hopper/angular_logs_ours_1.npy")
ang_disg = np.array([np.load("baselines/logs_20trainsteps_bs32/disag_hopper/dim_0_seed_{i}.npy".format(i=i)) for i in range(5)])
ang_cntb = np.array([np.load("baselines/logs/count_hopper/dim_0_seed_{i}.npy".format(i=i)) for i in range(5)])

acr = ang_rand > 0
acr = (acr/ang_rand.shape[-1]).sum(axis=-1)[:, :-51]

aco = ang_ours > 0
aco = (aco/ang_rand.shape[-1]).sum(axis=-1)[:, :-51]

acd = ang_disg > 0
acd = (acd/ang_disg.shape[-1]).sum(axis=-1)[:, :450]

acc = ang_cntb > 0
acc = (acc/ang_cntb.shape[-1]).sum(axis=-1)[:, :-39]

setup_plot()
plt.plot(x, np.mean(acr, axis=0), color=grey, label="Random")
plt.plot(x, np.mean(aco, axis=0), color=orange, label="Ours")
plt.plot(x, np.mean(acd, axis=0), color=teal, label="RL-Dis")
plt.plot(x, np.mean(acc, axis=0), color=purple, label="RL-Counts")

plt.fill_between(x, np.mean(acr, axis=0) - np.std(acr, axis=0) / np.sqrt(len(acr)),
                                 np.mean(acr, axis=0) + np.std(acr, axis=0) / np.sqrt(len(acr)),
                                 color=grey, alpha=0.1)
plt.fill_between(x, np.mean(aco, axis=0) - np.std(aco, axis=0) / np.sqrt(len(aco)),
                                 np.mean(aco, axis=0) + np.std(aco, axis=0) / np.sqrt(len(aco)),
                                 color=orange, alpha=0.1)
plt.fill_between(x, np.mean(acd, axis=0) - np.std(acd, axis=0) / np.sqrt(len(acd)),
                                 np.mean(acd, axis=0) + np.std(acd, axis=0) / np.sqrt(len(acd)),
                                 color=teal, alpha=0.1)
plt.fill_between(x, np.mean(acc, axis=0) - np.std(acc, axis=0) / np.sqrt(len(acc)),
                                 np.mean(acc, axis=0) + np.std(acc, axis=0) / np.sqrt(len(acc)),
                                 color=purple, alpha=0.1)
plt.plot(x, np.ones(len(x)), color=green,)
plt.legend()
plt.title("$\\theta_1$ Coverage")
plt.xlabel("Num Env. Steps")


plt.savefig("plots/theta_1_cov_hopper.pdf", bbox_inches='tight')


ang_rand = np.load("hopper/angular_logs_random_2.npy")
ang_ours = np.load("hopper/angular_logs_ours_2.npy")
ang_disg = np.array([np.load("baselines/logs_20trainsteps_bs32/disag_hopper/dim_1_seed_{i}.npy".format(i=i)) for i in range(5)])
ang_cntb = np.array([np.load("baselines/logs/count_hopper/dim_1_seed_{i}.npy".format(i=i)) for i in range(5)])

acr = ang_rand > 0
acr = (acr/ang_rand.shape[-1]).sum(axis=-1)[:, :-51]

aco = ang_ours > 0
aco = (aco/ang_rand.shape[-1]).sum(axis=-1)[:, :-51]

acd = ang_disg > 0
acd = (acd/ang_disg.shape[-1]).sum(axis=-1)[:, :450]

acc = ang_cntb > 0
acc = (acc/ang_cntb.shape[-1]).sum(axis=-1)[:, :-39]

setup_plot()
plt.plot(x, np.mean(acr, axis=0), color=grey, label="Random")
plt.plot(x, np.mean(aco, axis=0), color=orange, label="Ours")
plt.plot(x, np.mean(acd, axis=0), color=teal, label="RL-Dis")
plt.plot(x, np.mean(acc, axis=0), color=purple, label="RL-Counts")

plt.fill_between(x, np.mean(acr, axis=0) - np.std(acr, axis=0) / np.sqrt(len(acr)),
                                 np.mean(acr, axis=0) + np.std(acr, axis=0) / np.sqrt(len(acr)),
                                 color=grey, alpha=0.1)
plt.fill_between(x, np.mean(aco, axis=0) - np.std(aco, axis=0) / np.sqrt(len(aco)),
                                 np.mean(aco, axis=0) + np.std(aco, axis=0) / np.sqrt(len(aco)),
                                 color=orange, alpha=0.1)
plt.fill_between(x, np.mean(acd, axis=0) - np.std(acd, axis=0) / np.sqrt(len(acd)),
                                 np.mean(acd, axis=0) + np.std(acd, axis=0) / np.sqrt(len(acd)),
                                 color=teal, alpha=0.1)
plt.fill_between(x, np.mean(acc, axis=0) - np.std(acc, axis=0) / np.sqrt(len(acc)),
                                 np.mean(acc, axis=0) + np.std(acc, axis=0) / np.sqrt(len(acc)),
                                 color=purple, alpha=0.1)
plt.plot(x, np.ones(len(x)), color=green,)
plt.legend()
plt.title("$\\theta_2$ Coverage")
plt.xlabel("Num Env. Steps")
plt.savefig("plots/theta_2_cov_hopper.pdf", bbox_inches='tight')

ang_rand = np.load("hopper/angular_logs_random_3.npy")
ang_ours = np.load("hopper/angular_logs_ours_3.npy")
ang_disg = np.array([np.load("baselines/logs_20trainsteps_bs32/disag_hopper/dim_2_seed_{i}.npy".format(i=i)) for i in range(5)])
ang_cntb = np.array([np.load("baselines/logs/count_hopper/dim_2_seed_{i}.npy".format(i=i)) for i in range(5)])

acr = ang_rand > 0

acr = (acr/ang_rand.shape[-1]).sum(axis=-1)[:, :-51]

aco = ang_ours > 0
aco = (aco/ang_rand.shape[-1]).sum(axis=-1)[:, :-51]

acd = ang_disg > 0
acd = (acd/ang_disg.shape[-1]).sum(axis=-1)[:, :450]

acc = ang_cntb > 0
acc = (acc/ang_cntb.shape[-1]).sum(axis=-1)[:, :-39]

setup_plot()
plt.plot(x, np.mean(acr, axis=0), color=grey, label="Random")
plt.plot(x, np.mean(aco, axis=0), color=orange, label="Ours")
plt.plot(x, np.mean(acd, axis=0), color=teal, label="RL-Dis")
plt.plot(x, np.mean(acc, axis=0), color=purple, label="RL-Counts")

plt.fill_between(x, np.mean(acr, axis=0) - np.std(acr, axis=0) / np.sqrt(len(acr)),
                                 np.mean(acr, axis=0) + np.std(acr, axis=0) / np.sqrt(len(acr)),
                                 color=grey, alpha=0.1)
plt.fill_between(x, np.mean(aco, axis=0) - np.std(aco, axis=0) / np.sqrt(len(aco)),
                                 np.mean(aco, axis=0) + np.std(aco, axis=0) / np.sqrt(len(aco)),
                                 color=orange, alpha=0.1)
plt.fill_between(x, np.mean(acd, axis=0) - np.std(acd, axis=0) / np.sqrt(len(acd)),
                                 np.mean(acd, axis=0) + np.std(acd, axis=0) / np.sqrt(len(acd)),
                                 color=teal, alpha=0.1)
plt.fill_between(x, np.mean(acc, axis=0) - np.std(acc, axis=0) / np.sqrt(len(acc)),
                                 np.mean(acc, axis=0) + np.std(acc, axis=0) / np.sqrt(len(acc)),
                                 color=purple, alpha=0.1)
plt.plot(x, np.ones(len(x)), color=green,)
plt.legend()
plt.title("$\\theta_3$ Coverage")
plt.xlabel("Num Env. Steps")
plt.savefig("plots/theta_3_cov_hopper.pdf", bbox_inches='tight')

ang_rand = np.load("hopper/angular_logs_random_4.npy")
ang_ours = np.load("hopper/angular_logs_ours_4.npy")
ang_disg = np.array([np.load("baselines/logs_20trainsteps_bs32/disag_hopper/dim_3_seed_{i}.npy".format(i=i)) for i in range(5)])
ang_cntb = np.array([np.load("baselines/logs/count_hopper/dim_3_seed_{i}.npy".format(i=i)) for i in range(5)])

acr = ang_rand > 0
acr = (acr/ang_rand.shape[-1]).sum(axis=-1)[:, :-51]

aco = ang_ours > 0
aco = (aco/ang_rand.shape[-1]).sum(axis=-1)[:, :-51]

acd = ang_disg > 0
acd = (acd/ang_disg.shape[-1]).sum(axis=-1)[:, :450]

acc = ang_cntb > 0
acc = (acc/ang_cntb.shape[-1]).sum(axis=-1)[:, :-39]

setup_plot()
plt.plot(x, np.mean(acr, axis=0), color=grey, label="Random")
plt.plot(x, np.mean(aco, axis=0), color=orange, label="Ours")
plt.plot(x, np.mean(acd, axis=0), color=teal, label="RL-Dis")
plt.plot(x, np.mean(acc, axis=0), color=purple, label="RL-Counts")

plt.fill_between(x, np.mean(acr, axis=0) - np.std(acr, axis=0) / np.sqrt(len(acr)),
                                 np.mean(acr, axis=0) + np.std(acr, axis=0) / np.sqrt(len(acr)),
                                 color=grey, alpha=0.1)
plt.fill_between(x, np.mean(aco, axis=0) - np.std(aco, axis=0) / np.sqrt(len(aco)),
                                 np.mean(aco, axis=0) + np.std(aco, axis=0) / np.sqrt(len(aco)),
                                 color=orange, alpha=0.1)
plt.fill_between(x, np.mean(acd, axis=0) - np.std(acd, axis=0) / np.sqrt(len(acd)),
                                 np.mean(acd, axis=0) + np.std(acd, axis=0) / np.sqrt(len(acd)),
                                 color=teal, alpha=0.1)
plt.fill_between(x, np.mean(acc, axis=0) - np.std(acc, axis=0) / np.sqrt(len(acc)),
                                 np.mean(acc, axis=0) + np.std(acc, axis=0) / np.sqrt(len(acc)),
                                 color=purple, alpha=0.1)
plt.plot(x, np.ones(len(x)), color=green,)
plt.legend()
plt.title("$\\theta_4$ Coverage")
plt.xlabel("Num Env. Steps")
plt.savefig("plots/theta_4_cov_hopper.pdf", bbox_inches='tight')
