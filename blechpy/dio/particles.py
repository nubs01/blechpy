import tables

class trial_info_particle(tables.IsDescription):
    '''PyTables particle for recording digital input (taste) trial info/order
    '''
    trial_num = tables.UInt16Col()
    channel = tables.Int16Col()
    name = tables.StringCol(20)
