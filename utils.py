from wrapper_envs import *
def get_env_expldim_bins(name):
    """ 
    Returns a tuple of (env, expl_dims, bins), where env is the environment, 
    expl_dims is a list of the dimensions of the observations space that are considered for exploration,
    and bins is a list of the corresponding bins for each of the exploration dimensions.
    """

    if name == 'inverted_pendulum':
        return gym.make("InvertedPendulum-v2"), [0,1], [np.arange(-1.15, 1.15, 0.01),
                                                         np.arange(-1.75, 1.75, 0.01),
                                                         np.arange(-7, 7, 0.01),
                                                         np.arange(-10, 10, 0.01)]

    elif name == 'inverted_double_pendulum':
        return  InvertedDoublePendulum(), [0,1,2], [np.arange(-1.1, 1.1, 0.01),
                                                    np.arange(0, np.pi, 0.01),
                                                    np.arange(0, np.pi, 0.01),
                                                   np.arange(-1, 10, 0.01),
                                                   np.arange(-1, 10, 0.01),
                                                   np.arange(-1, 10, 0.01),
                                                    np.arange(-1, 10, 0.01),
                                                     np.arange(0, 0.02, 0.01),
                                                      np.arange(0, 0.02, 0.01)]

    elif name == 'hopper':

        return HopperEnv(), [0,1,2,3],   [np.arange(-1.3, 1.3, 0.01),
                                         np.arange(0, 1.5, 0.01),
                                         np.arange(-5.3, 1.100, 0.01),
                                         np.arange(-2.8, 0.1, 0.01),
                                        np.arange(-2.8, 0.1, 0.01),
                                        np.arange(-0.9, 0.9, 0.01),
                                        np.arange(-3 , 3, 0.01),
                                        np.arange(-7, 2.5, 0.01),
                                        np.arange(-10., 10, 0.01),
                                        np.arange(-10, 10, 0.01),
                                        np.arange(-10, 10, 0.01),
                                        np.arange( -10, 10, 0.01)]
    else:
        raise AssertionError('env name ' + str(name) + ' not found')