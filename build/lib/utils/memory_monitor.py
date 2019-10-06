import os
import psutil
import resource
import inspect
import sys

def get_location():
  (frame, filename, line_number,
     function_name, lines, index) = inspect.getouterframes(inspect.currentframe())[2]
  return filename , function_name ,line_number       


def memory_usage_psutil():
    process = psutil.Process(os.getpid())
    current_mem = process.get_ext_memory_info()
    MB_conv = 2**20
    return current_mem.rss / MB_conv , current_mem.vms / MB_conv , current_mem.shared / MB_conv 

def memory_usage_resource():
    MB_conv = 2**10
    max_mem  = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / MB_conv
    return max_mem

def report_usage(label="", logfile = sys.stderr):
    myfile , myfunction , myline = get_location()
    maxmem = memory_usage_resource()
    current_mem  = memory_usage_psutil()
    print ( "Location %s: file = %s , function = %s , line = %s" %   (label , myfile , myfunction , myline) , file = logfile)
    print ( "\trss_mem = %lf MB ,vms_mem = %lf MB ,shared_mem = %lf MB, max_mem = %lf MB" %  (  current_mem + ( maxmem, )) , file = logfile )
    return

