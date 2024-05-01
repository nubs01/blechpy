import blechpy
from blechpy.analysis import poissonHMM as phmm

rec_dir = '/media/dsvedberg/Ubuntu Disk/TE_sandbox/DS39_spont_taste_201029_154308'

dat = blechpy.load_dataset(rec_dir)

handler = phmm.HmmHandler(dat)

params = [{'threshold':0.1E-15,'max_iter': 200, 'n_repeats':5, 'time_start':-250,'time_end': 2500, 'dt': 0.01, 'n_states': x,'area': 'GC', 'taste':['Suc', 'NaCl', 'CA', 'QHCl']} for x in [10]]

handler.add_params(params)

handler.run(constraint_func = phmm.sequential_constraint, n_cpu = 6)

handler.plot_saved_models()