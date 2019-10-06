import sys
import time
from functools import wraps

def Logger(heading):
    def real_logger(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            old_out = sys.stdout
            sys.stdout.write(heading+'...')
            sys.stdout.flush()
            if hasattr(args[0], 'log_file'):
                print('')
                log_file = args[0].log_file
                with open(log_file, 'a') as f:
                    sys.stdout = f
                    try:
                        func(*args, **kwargs)
                        fail = False
                    except Exception as e:
                        sys.__stdout__.write('\n\nException in %s.%s\n\n' % (func.__module__, func.__name__))
                        fail = e

                    sys.stdout = old_out
                    if fail is not False:
                        raise fail

            else:
                func(*args, **kwargs)
                print('Done!\n')

        return wrapper
    return real_logger


def Timer(heading):
    def real_timer(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            print('')
            print('----------\n%s\n----------' % heading)
            result = func(*args, **kwargs)
            print('Done! Elapsed Time: %1.2f\n' % (time.time()-start))
            return result
        return wrapper
    return real_timer
