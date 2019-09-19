import sys
import time

def Logger(heading):
    def real_logger(func):
        def wrapper(*args, **kwargs):
            old_out = sys.stdout
            sys.stdout.write(heading+'...')
            sys.stdout.flush()
            if hasattr(args[0], 'log_file'):
                log_file = args[0].log_file
                with open(log_file, 'a') as f:
                    sys.stdout = f
                    func(*args, **kwargs)
                    sys.stdout = old_out
            else:
                func(*args, **kwargs)
                print('Done!')

        return wrapper
    return real_logger


def Timer(heading):
    def real_timer(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            print('')
            print('----------\n%s\n----------' % heading)
            result = func(*args, **kwargs)
            print('Done! Elapsed Time: %1.2f' % (time.time()-start))
            return result
        return wrapper
    return real_timer
