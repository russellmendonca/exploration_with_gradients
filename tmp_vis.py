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


def trial_plot_for_legend():
    setup_plot()
   
    plt.plot(x, x, color=orange, label="Ours")
    plt.plot(x, x, color=teal, label="RL-Dis")
    plt.plot(x, x, color=purple, label="RL-Counts")
    plt.plot(x, x, color=grey, label="Random")
    plt.title('trial')
    plt.legend(ncol = 4, bbox_to_anchor=[-0.1, -0.5])
    plt.savefig( "plots/legend.pdf", bbox_inches='tight')


def gen_entropy_plot(name, title, rand, ours, disg, cntb):
    
    ler = np.zeros_like(rand[:, :, 0])
    for i in range(len(ler)):
        for j in range(len(ler[i])):
            v = rand[i][j] / np.sum(rand[i][j])
            ler[i, j] = np.sum(-np.log(v + 1e-8) * v)


    leo = np.zeros_like(ours[:, :, 0])
    for i in range(len(leo)):
        for j in range(len(leo[i])):
            v = ours[i][j] / np.sum(ours[i][j])
            leo[i, j] = np.sum(-np.log(v + 1e-8) * v)


    led = np.zeros_like(disg[:, :, 0])
    for i in range(len(led)):
        for j in range(len(led[i])):
            v = disg[i][j] / np.sum(disg[i][j])
            led[i, j] = np.sum(-np.log(v + 1e-8) * v)

    lec = np.zeros_like(cntb[:, :, 0])
    for i in range(len(lec)):
        for j in range(len(lec[i])):
            v = cntb[i][j] / np.sum(cntb[i][j])
            lec[i, j] = np.sum(-np.log(v + 1e-8) * v)


    
    ler = ler[:,:450]
    leo = leo[:,:450]
    led = led[:,:450]
    lec = lec[:,:450]
    setup_plot()
    plt.plot(x, np.mean(ler, axis=0), color=grey, label="Random")
    plt.plot(x, np.mean(leo, axis=0), color=orange, label="Ours")
    plt.plot(x, np.mean(led, axis=0), color=teal, label="RL-Dis")
    plt.plot(x, np.mean(lec, axis=0), color=purple, label="RL-Counts")

    plt.fill_between(x, np.mean(ler, axis=0) - np.std(ler, axis=0) / np.sqrt(len(ler)),
                                    np.mean(ler, axis=0) + np.std(ler, axis=0) / np.sqrt(len(ler)),
                                    color=grey, alpha=0.1)
    plt.fill_between(x, np.mean(leo, axis=0) - np.std(leo, axis=0) / np.sqrt(len(leo)),
                                    np.mean(leo, axis=0) + np.std(leo, axis=0) / np.sqrt(len(leo)),
                                    color=orange, alpha=0.1)
    plt.fill_between(x, np.mean(led, axis=0) - np.std(led, axis=0) / np.sqrt(len(led)),
                                    np.mean(led, axis=0) + np.std(led, axis=0) / np.sqrt(len(led)),
                                    color=teal, alpha=0.1)
    plt.fill_between(x, np.mean(lec, axis=0) - np.std(lec, axis=0) / np.sqrt(len(lec)),
                                    np.mean(lec, axis=0) + np.std(lec, axis=0) / np.sqrt(len(lec)),
                                    color=purple, alpha=0.1)
    #plt.plot(x, np.ones(101) * np.log(rand.shape[-1]), color=green,)
    #plt.legend(loc="lower right")
    plt.title(title)
    plt.xlabel("Num Env. Steps")
    plt.savefig( name + ".pdf", bbox_inches='tight')


def gen_cov_plot(name, title, rand, ours, disg, cntb):
    
    acr = rand > 0
    acr = (acr/rand.shape[-1]).sum(axis=-1)[:, :450]

    aco = ours > 0
    aco = (aco/rand.shape[-1]).sum(axis=-1)[:, :450]

    acd = disg > 0
    acd = (acd/disg.shape[-1]).sum(axis=-1)[:, :450]

    acc = cntb > 0
    acc = (acc/cntb.shape[-1]).sum(axis=-1)[:, :450]

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
    #plt.legend(loc="lower right")
    plt.title(title)
    plt.xlabel("Num Env. Steps")
    plt.savefig( name + ".pdf", bbox_inches='tight')

def gen_plots(name, title, rand, ours, disg, cntb):

    gen_entropy_plot('plots/entropy/' + name, title + " Entropy", rand, ours, disg, cntb )
    gen_cov_plot('plots/coverage/' + name, title + " Coverage", rand, ours, disg, cntb )




trial_plot_for_legend()
#ip 
lin_rand = np.load("inverted_pendulum/linear_logs_random.npy")
lin_ours = np.load("inverted_pendulum/linear_logs_ours.npy")
lin_disg = np.array([np.load("baselines/logs_20trainsteps_bs32/disag_inverted_pendulum/dim_0_seed_{i}.npy".format(i=i)) for i in range(5)])
lin_cntb = np.array([np.load("baselines/logs/count_inverted_pendulum/dim_0_seed_{i}.npy".format(i=i)) for i in range(5)])

gen_plots('ip_lin', "$x$", lin_rand, lin_ours, lin_disg, lin_cntb)

ang_rand = np.load("inverted_pendulum/angular_logs_random.npy")
ang_ours = np.load("inverted_pendulum/angular_logs_ours.npy")
ang_disg = np.array([np.load("baselines/logs_20trainsteps_bs32/disag_inverted_pendulum/dim_1_seed_{i}.npy".format(i=i)) for i in range(5)])
ang_cntb = np.array([np.load("baselines/logs/count_inverted_pendulum/dim_1_seed_{i}.npy".format(i=i)) for i in range(5)])

gen_plots('ip_ang', "$\\theta$", ang_rand, ang_ours, ang_disg, ang_cntb)


#hopper

for dim in range(4):
    rand = np.load("hopper/angular_logs_random_"+str(dim+1)+".npy")
    ours = np.load("hopper/angular_logs_ours_" + str(dim+1) + ".npy")
    disg = np.array([np.load("baselines/logs_20trainsteps_bs32/disag_hopper/dim_"+str(dim) + "_seed_{i}.npy".format(i=i)) for i in range(5)])
    cntb = np.array([np.load("baselines/logs/count_hopper/dim_"+str(dim) + "_seed_{i}.npy".format(i=i)) for i in range(5)])
    gen_plots('hopper_dim'+str(dim), "$\\theta_"+str(dim+1)+"$",  rand, ours, disg, cntb)



#double ip
lin_rand = np.load("inverted_double_pendulum/linear_logs_random.npy")
lin_ours = np.load("inverted_double_pendulum/linear_logs_ours.npy")
lin_disg = np.array([np.load("baselines/logs_20trainsteps_bs32/disag_inverted_double_pendulum/dim_0_seed_{i}.npy".format(i=i)) for i in range(5)])
lin_cntb = np.array([np.load("baselines/logs/count_inverted_double_pendulum/dim_0_seed_{i}.npy".format(i=i)) for i in range(5)])

gen_plots('dip_lin', "$x$", lin_rand, lin_ours, lin_disg, lin_cntb)

ang_rand = np.load("inverted_double_pendulum/angular_logs_random_1.npy")
ang_ours = np.load("inverted_double_pendulum/angular_logs_ours_1.npy")
ang_disg = np.array([np.load("baselines/logs_20trainsteps_bs32/disag_inverted_double_pendulum/dim_1_seed_{i}.npy".format(i=i)) for i in range(5)])
ang_cntb = np.array([np.load("baselines/logs/count_inverted_double_pendulum/dim_1_seed_{i}.npy".format(i=i)) for i in range(5)])

gen_plots('dip_ang1',"$\\theta_1$",  ang_rand, ang_ours, ang_disg, ang_cntb)

ang_rand = np.load("inverted_double_pendulum/angular_logs_random_2.npy")
ang_ours = np.load("inverted_double_pendulum/angular_logs_ours_2.npy")
ang_disg = np.array([np.load("baselines/logs_20trainsteps_bs32/disag_inverted_double_pendulum/dim_2_seed_{i}.npy".format(i=i)) for i in range(5)])
ang_cntb = np.array([np.load("baselines/logs/count_inverted_double_pendulum/dim_2_seed_{i}.npy".format(i=i)) for i in range(5)])

gen_plots('dip_ang2', "$\\theta_2$", ang_rand, ang_ours, ang_disg, ang_cntb)
