import tables

class trial_info_particle(tables.IsDescription):
    '''PyTables particle for recording digital input (taste) trial info/order
    '''
    trial_num = tables.UInt16Col()
    channel = tables.Int16Col()
    name = tables.StringCol(20)


class unit_descriptor(tables.IsDescription):
    '''PyTables particles for storing sorted unit information 
    '''
    electrode_number = tables.Int32Col()
    single_unit = tables.Int32Col()
    regular_spiking = tables.Int32Col()
    fast_spiking = tables.Int32Col()
