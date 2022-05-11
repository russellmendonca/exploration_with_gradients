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

lin_rand = np.load("inverted_double_pendulum/linear_logs_random.npy")
lin_ours = np.load("inverted_double_pendulum/linear_logs_ours.npy")
lin_disg = np.array([np.load("baselines/logs_20trainsteps_bs32/disag_inverted_double_pendulum/dim_0_seed_{i}.npy".format(i=i)) for i in range(5)])
lin_cntb = np.array([np.load("baselines/logs/count_inverted_double_pendulum/dim_0_seed_{i}.npy".format(i=i)) for i in range(5)])

x = (5 * np.arange(450)) * 100


lcr = lin_rand > 0
lcr = (lcr/lin_rand.shape[-1]).sum(axis=-1)[:, :-51]

lco = lin_ours > 0
lco = (lco/lin_rand.shape[-1]).sum(axis=-1)[:, :-51]
lcd = lin_disg > 0
lcd = (lcd/lin_disg.shape[-1]).sum(axis=-1)[:, :450]

lcc = lin_cntb > 0
lcc = (lcc/lin_cntb.shape[-1]).sum(axis=-1)[:, :-39]


setup_plot()
plt.plot(x, np.mean(lcr, axis=0), color=grey, label="Random")
plt.plot(x, np.mean(lco, axis=0), color=orange, label="Ours")
plt.plot(x, np.mean(lcd, axis=0), color=teal, label="RL-Dis")
plt.plot(x, np.mean(lcc, axis=0), color=purple, label="RL-Counts")

plt.fill_between(x, np.mean(lcr, axis=0) - np.std(lcr, axis=0) / np.sqrt(len(lcr)),
                                 np.mean(lcr, axis=0) + np.std(lcr, axis=0) / np.sqrt(len(lcr)),
                                 color=grey, alpha=0.1)
plt.fill_between(x, np.mean(lco, axis=0) - np.std(lco, axis=0) / np.sqrt(len(lco)),
                                 np.mean(lco, axis=0) + np.std(lco, axis=0) / np.sqrt(len(lco)),
                                 color=orange, alpha=0.1)
plt.fill_between(x, np.mean(lcd, axis=0) - np.std(lcd, axis=0) / np.sqrt(len(lcd)),
                                 np.mean(lcd, axis=0) + np.std(lcd, axis=0) / np.sqrt(len(lcd)),
                                 color=teal, alpha=0.1)
plt.fill_between(x, np.mean(lcc, axis=0) - np.std(lcc, axis=0) / np.sqrt(len(lcc)),
                                 np.mean(lcc, axis=0) + np.std(lcc, axis=0) / np.sqrt(len(lcc)),
                                 color=purple, alpha=0.1)
plt.plot(x, np.ones(len(x)), color=green,)
plt.legend(ncol=2)
plt.title("$x$ Coverage")
plt.xlabel("Num Env. Steps")
plt.savefig("plots/x_cov_idp.pdf", bbox_inches='tight')


# ler = np.zeros_like(lin_rand[:, :, 0])
# for i in range(len(ler)):
#     for j in range(len(ler[i])):
#         v = lin_rand[i][j] / np.sum(lin_rand[i][j])
#         ler[i, j] = np.sum(-np.log(v + 1e-8) * v)


# leo = np.zeros_like(lin_ours[:, :, 0])
# for i in range(len(leo)):
#     for j in range(len(leo[i])):
#         v = lin_ours[i][j] / np.sum(lin_ours[i][j])
#         leo[i, j] = np.sum(-np.log(v + 1e-8) * v)


# led = np.zeros_like(lin_disg[:, :, 0])
# for i in range(len(led)):
#     for j in range(len(led[i])):
#         v = lin_disg[i][j] / np.sum(lin_disg[i][j])
#         led[i, j] = np.sum(-np.log(v + 1e-8) * v)

# lec = np.zeros_like(lin_cntb[:, :, 0])
# for i in range(len(lec)):
#     for j in range(len(lec[i])):
#         v = lin_cntb[i][j] / np.sum(lin_cntb[i][j])
#         lec[i, j] = np.sum(-np.log(v + 1e-8) * v)


# setup_plot()
# plt.plot(x, np.mean(ler, axis=0), color=grey, label="Random")
# plt.plot(x, np.mean(leo, axis=0), color=orange, label="Ours")
# plt.plot(x, np.mean(led, axis=0), color=teal, label="RL-Dis")
# plt.plot(x, np.mean(lec, axis=0), color=purple, label="RL-Counts")

# plt.fill_between(x, np.mean(ler, axis=0) - np.std(ler, axis=0) / np.sqrt(len(ler)),
#                                  np.mean(ler, axis=0) + np.std(ler, axis=0) / np.sqrt(len(ler)),
#                                  color=grey, alpha=0.1)
# plt.fill_between(x, np.mean(leo, axis=0) - np.std(leo, axis=0) / np.sqrt(len(leo)),
#                                  np.mean(leo, axis=0) + np.std(leo, axis=0) / np.sqrt(len(leo)),
#                                  color=orange, alpha=0.1)
# plt.fill_between(x, np.mean(led, axis=0) - np.std(led, axis=0) / np.sqrt(len(led)),
#                                  np.mean(led, axis=0) + np.std(led, axis=0) / np.sqrt(len(led)),
#                                  color=teal, alpha=0.1)
# plt.fill_between(x, np.mean(lec, axis=0) - np.std(lec, axis=0) / np.sqrt(len(lec)),
#                                  np.mean(lec, axis=0) + np.std(lec, axis=0) / np.sqrt(len(lec)),
#                                  color=purple, alpha=0.1)
# plt.plot(x, np.ones(101) * np.log(lin_rand.shape[-1]), color=green,)
# plt.legend(loc="lower right")
# plt.title("$x$ Visitation Entropy")
# plt.xlabel("Num Env. Steps")
# plt.savefig("x_ent.pdf", bbox_inches='tight')


ang_rand = np.load("inverted_double_pendulum/angular_logs_random_1.npy")
ang_ours = np.load("inverted_double_pendulum/angular_logs_ours_1.npy")
ang_disg = np.array([np.load("baselines/logs/disag_inverted_double_pendulum/dim_1_seed_{i}.npy".format(i=i)) for i in range(5)])
ang_cntb = np.array([np.load("baselines/logs/count_inverted_double_pendulum/dim_1_seed_{i}.npy".format(i=i)) for i in range(5)])

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
plt.savefig("plots/theta_1_cov_idp.pdf", bbox_inches='tight')


# aer = np.zeros_like(ang_rand[:, :, 0])
# for i in range(len(aer)):
#     for j in range(len(aer[i])):
#         v = ang_rand[i][j] / np.sum(ang_rand[i][j])
#         aer[i, j] = np.sum(-np.log(v + 1e-8) * v)

# aeo = np.zeros_like(ang_ours[:, :, 0])
# for i in range(len(aeo)):
#     for j in range(len(aeo[i])):
#         v = ang_ours[i][j] / np.sum(ang_ours[i][j])
#         aeo[i, j] = np.sum(-np.log(v + 1e-8) * v)

# aed = np.zeros_like(ang_disg[:, :, 0])
# for i in range(len(aed)):
#     for j in range(len(aed[i])):
#         v = ang_disg[i][j] / np.sum(ang_disg[i][j])
#         aed[i, j] = np.sum(-np.log(v + 1e-8) * v)

# aec = np.zeros_like(ang_cntb[:, :, 0])
# for i in range(len(aec)):
#     for j in range(len(aec[i])):
#         v = ang_cntb[i][j] / np.sum(ang_cntb[i][j])
#         aec[i, j] = np.sum(-np.log(v + 1e-8) * v)

# setup_plot()
# plt.plot(x, np.mean(aer, axis=0), color=grey, label="Random")
# plt.plot(x, np.mean(aeo, axis=0), color=orange, label="Ours")
# plt.plot(x, np.mean(aed, axis=0), color=teal, label="RL-Dis")
# plt.plot(x, np.mean(aec, axis=0), color=purple, label="RL-Counts")

# plt.fill_between(x, np.mean(aer, axis=0) - np.std(aer, axis=0) / np.sqrt(len(aer)),
#                                  np.mean(aer, axis=0) + np.std(aer, axis=0) / np.sqrt(len(aer)),
#                                  color=grey, alpha=0.1)
# plt.fill_between(x, np.mean(aeo, axis=0) - np.std(aeo, axis=0) / np.sqrt(len(aeo)),
#                                  np.mean(aeo, axis=0) + np.std(aeo, axis=0) / np.sqrt(len(aeo)),
#                                  color=orange, alpha=0.1)
# plt.fill_between(x, np.mean(aed, axis=0) - np.std(aed, axis=0) / np.sqrt(len(aed)),
#                                  np.mean(aed, axis=0) + np.std(aed, axis=0) / np.sqrt(len(aed)),
#                                  color=teal, alpha=0.1)
# plt.fill_between(x, np.mean(aec, axis=0) - np.std(aec, axis=0) / np.sqrt(len(aec)),
#                                  np.mean(aec, axis=0) + np.std(aec, axis=0) / np.sqrt(len(aec)),
#                                  color=purple, alpha=0.1)


# plt.plot(x, np.ones(101) * np.log(ang_rand.shape[-1]), color=green,)
# plt.legend(loc="center right")
# plt.title("$\\theta$ Visitation Entropy")
# plt.xlabel("Num Env. Steps")
# plt.savefig("theta_ent.pdf", bbox_inches='tight')

ang_rand = np.load("inverted_double_pendulum/angular_logs_random_2.npy")
ang_ours = np.load("inverted_double_pendulum/angular_logs_ours_2.npy")
ang_disg = np.array([np.load("baselines/logs_20trainsteps_bs32/disag_inverted_double_pendulum/dim_2_seed_{i}.npy".format(i=i)) for i in range(5)])
ang_cntb = np.array([np.load("baselines/logs/count_inverted_double_pendulum/dim_2_seed_{i}.npy".format(i=i)) for i in range(5)])

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
plt.savefig("plots/theta_2_cov_idp.pdf", bbox_inches='tight')
