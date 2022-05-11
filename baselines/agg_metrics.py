
import pickle
import numpy as np

def agg_across_seeds(exp_dir, dims, num_seeds=5):
    
    for dim in dims:
        data_across_seeds = []
        for seed in range(num_seeds):   
            _file = exp_dir + 'dim_' + str(dim) + '_seed_' +str(seed) + '.npy'
            data_across_seeds.append(np.load(_file))
        
        np.save(exp_dir + 'agg_seed_dim_' + str(dim), data_across_seeds)
      

agg_across_seeds('logs/count_hopper/', dims=[0,1,2,3], num_seeds=5) 
agg_across_seeds('logs/count_inverted_pendulum/', dims=[0,1], num_seeds=5)
agg_across_seeds('logs/count_inverted_double_pendulum/', dims=[0,1,2], num_seeds=5)


#agg_across_seeds('logs/disag_hopper/', dims=[0,1,2,3], num_seeds=5) 
#agg_across_seeds('logs/disag_inverted_pendulum/', dims=[0,1], num_seeds=5)
#agg_across_seeds('logs/disag_inverted_double_pendulum/', dims=[0,1,2], num_seeds=5)
