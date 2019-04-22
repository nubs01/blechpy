import tables,os
import blech_clust.dio.blech_params as params

def create_empty_data_h5(filename):
    '''
    create empty h5 store for blech data
    '''
    if not filename.endswith('.h5'):
        filename+='.h5'
    hf5 = tables.open_file(filename,'w',title=filename.replace('.h5',''))
    hf5.create_group('/','raw')
    hf5.create_group('/', 'raw_emg')
    hf5.create_group('/', 'digital_in')
    hf5.create_group('/', 'digital_out')
    hf5.close()
    return filename

def get_h5_name(file_dir):
    '''
    return the name of the h5 file found in file_dir
    asks for selection if multiple found
    '''
    file_list = os.listdir(file_dir)
    h5_files = [f for f in file_list if f.endswith('.h5')]
    if len(h5_files)>1:
        choice = params.select_from_list('Choose which h5 file to load','Multiple h5 stores found',h5_files)
        if choice is None:
            return None
        else:
            h5_files = [choice]
    return h5_files[0]

def get_h5(file_dir):
    '''
    finds and opens the h5 file in file_dir, allows selection if multiple found
    returns tables file object with h5 data
    '''
    h5_file = get_h5_name(file_dir)
    hf5 = tables.open_file(os.path.join(file_dir,h5_file),'r+')
    return hf5
