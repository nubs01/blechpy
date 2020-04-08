def check(obj, attr, value=None):
    if hasattr(obj, attr):
        if callable(value):
            return value(out[attr])
        elif value is None:
            return True
        else:
            return out[attr] == value
    else:
        return False


def validate_data_object(dat):
    attrs = ['root_dir', 'data_type', 'data_name', 'save_file', 'log_file']
    values = [os.path.isdir, None, None, os.path.isfile, os.path.isfile]
    out = {x: check(dat, x, y) for x, y in zip(attrs, values)}
    return out


def fix_dataset(dat):
    out = validate_data_object(dat)

    attrs = ['h5_file', 'dataset_creation_date', 'processing_steps',
             'processing_status', 'sample_rate', 'rec_info', 'dig_in_mapping',
             'dig_out_mapping', 'electrode_mapping', ]
    steps = blechpy.dataset.PROCESSING_STEPS.copy()

    # Check that the right attributes exist
    pass

