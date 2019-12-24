import os
import shutil

def reset_directory(file_dir, keep_pickle=True, keep_params=True):
    keepers = ['.rhd','.dat']
    if keep_pickle:
        keepers.append('.p')

    if keep_params:
        keepers.append('analysis_params')
        keepers.append('.json')

    file_list = os.listdir(file_dir)
    for fn in file_list:
        tmp = os.path.join(file_dir, fn)
        if any([fn.endswith(x) for x in keepers]):
            continue
        else:
            if os.path.isfile(tmp):
                os.remove(tmp)
            elif os.path.isdir(tmp):
                shutil.rmtree(tmp)


def fix_filenames(file_dir):
    if file_dir.endswith(os.sep):
        file_dir = file_dir[:-1]

    base = os.path.basename(file_dir)
    fixed = base[:-14]

    file_list = os.listdir(file_dir)
    for fn in file_list:
        if base in fn:
            tmp = os.path.join(file_dir, fn)
            new_tmp = tmp.replace(base, fixed)
            os.rename(tmp, new_tmp)

