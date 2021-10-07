import tables

class trial_info_particle(tables.IsDescription):
    '''PyTables particle for recording digital input (taste) trial info/order
    '''
    trial_num = tables.UInt16Col()
    channel = tables.Int16Col()
    name = tables.StringCol(20)
    on_index = tables.Int32Col()
    off_index = tables.Int32Col()
    on_time = tables.Float32Col()
    off_time = tables.Float32Col()


class unit_descriptor(tables.IsDescription):
    '''PyTables particles for storing sorted unit information 
    '''
    electrode_number = tables.Int32Col()
    single_unit = tables.Int32Col()
    regular_spiking = tables.Int32Col()
    fast_spiking = tables.Int32Col()


class electrode_map_particle(tables.IsDescription):
    '''PyTables particle for storing electrode mapping
    '''
    Electrode = tables.Int16Col()
    Port = tables.StringCol(5)
    Channel = tables.Int16Col()
    area = tables.StringCol(20)
    CAR_group = tables.Int16Col()
    dead = tables.BoolCol()
    cutoff_time = tables.Float32Col()
    clustering_result = tables.Int16Col()


class digital_mapping_particle(tables.IsDescription):
    '''Pytables particle for storing digital input/output mappings
    '''
    channel = tables.Int16Col()
    name = tables.StringCol(20)
    palatability_rank = tables.Int16Col()
    laser = tables.BoolCol()
    spike_array = tables.BoolCol()
    exclude = tables.BoolCol()
    laser_channels = tables.BoolCol()


class HMMInfoParticle(tables.IsDescription):
    #   HMM_ID, taste, din_channel, n_cells, time_start, time_end, thresh,
    #   unit_type, n_repeats, dt, n_states, n_iters, BIC, cost, converged,
    #   area
    hmm_id = tables.Int16Col()
    taste = tables.StringCol(45)
    channel = tables.Int16Col()
    n_cells = tables.Int32Col()
    unit_type = tables.StringCol(15)
    n_trials = tables.Int32Col()
    dt = tables.Float64Col()
    max_iter = tables.Int32Col()
    threshold = tables.Float64Col()
    time_start = tables.Int32Col()
    time_end = tables.Int32Col()
    n_repeats = tables.Int16Col()
    n_states = tables.Int32Col()
    n_iterations = tables.Int32Col()
    BIC = tables.Float64Col()
    cost = tables.Float64Col()
    converged = tables.BoolCol()
    fitted = tables.BoolCol()
    max_log_prob = tables.Float64Col()
    log_likelihood = tables.Float64Col()
    area = tables.StringCol(15)
    hmm_class = tables.StringCol(20)
    notes = tables.StringCol(40)

class AnonHMMInfoParticle(tables.IsDescription):
    #   HMM_ID, taste, din_channel, n_cells, time_start, time_end, thresh,
    #   unit_type, n_repeats, dt, n_states, n_iters, BIC, cost, converged,
    #   area
    # info particle for anonymous hmm data, so if hdf5 store isn't tied to a
    # single recording, adds column for rec_dir
    hmm_id = tables.Int16Col()
    taste = tables.StringCol(45)
    channel = tables.Int32Col()
    n_cells = tables.Int32Col()
    unit_type = tables.StringCol(15)
    n_trials = tables.Int32Col()
    dt = tables.Float64Col()
    max_iter = tables.Int32Col()
    threshold = tables.Float64Col()
    time_start = tables.Int32Col()
    time_end = tables.Int32Col()
    n_repeats = tables.Int16Col()
    n_states = tables.Int32Col()
    n_iterations = tables.Int32Col()
    BIC = tables.Float64Col()
    cost = tables.Float64Col()
    converged = tables.BoolCol()
    fitted = tables.BoolCol()
    max_log_prob = tables.Float64Col()
    log_likelihood = tables.Float64Col()
    area = tables.StringCol(15)
    hmm_class = tables.StringCol(20)
    notes = tables.StringCol(30)
    rec_dir = tables.StringCol(150)
