import pandas as pd
import datetime as dt
from copy import deepcopy
import re

def get_datetime_from_str(date_str,accepted_formats = ['%m/%d/%y','%m/%d/%y %H:%M',
                                                        '%m%d%y %H:%M','%H:%M']):
    '''
    check datetime string format and convert to datetime
    '''
    out = None
    if date_str is None:
        return None
    for fmt in accepted_formats:
        try:
            out = dt.datetime.strptime(date_str,fmt)
            if out != None:
                break
        except ValueError:
            out = None
    return out

def get_date_str(date,fmt='%m/%d/%y'):
    '''
    get string from datatime object. None object will return a default date
    string depending on the desired fmt. i.e. fmt='%m/%d/%y' will return
    'mm/dd/yy' for None input
    '''
    null_convert = {'%m':'mm','%d':'dd','%y':'yy','%H':'HH','%M':'MM'}
    if date is None or date is '':
        out = fmt
        for k,v in null_convert.items():
            out = out.replace(k,v)
        return out
    return date.strftime(fmt)
        

def print_dict(dic,tabs=0):
    '''
    Turns a dict into a string recursively
    '''
    dic = deepcopy(dic)
    if isinstance(dic,str) or isinstance(dic,int):
        out = str(dic)
        for i in range(tabs):
            out = '    '+out
        return out
    out = ''
    spacing = str(max([len(x) for x in dic.keys()])+4)
    for k,v in dic.items():
        if isinstance(v,pd.DataFrame):
            v_str = '\n'+print_dataframe(v,tabs+1)+'\n'
        elif isinstance(v,dict):
            v_str = '\n'+print_dict(v,tabs+1)
        elif isinstance(v,list):
            v_str = [print_dict(x,tabs) for x in v]
            v_str = '\n'+''.join(v_str)
            v_str = v_str.replace('\n','\n    ')
        elif isinstance(v,dt.datetime):
            if v.hour==0 and v.minute==0:
                v_str = v.strftime('%m/%d/%y')
            else:
                v_str = v.strftime('%m/%d/%y %H:%M')
        elif isinstance(v,dt.date):
            v_str = v.strftime('%m/%d/%y')
        else:
            v_str = v
        fmt = "{:<"+spacing+"}{}"
        out = out + fmt.format(k,v_str) + '\n'
    for i in range(tabs):
        out = '    '+out.replace('\n','\n    ')
    return out

def print_dataframe(df,tabs=0,idxfmt='Date'):
    '''
    Turns a pandas dataframe into a string without numerical index, date index
    will print and be formatted according to idxfmt Date, Datetime or Time
    '''
    df = df.copy()
    if df.empty:
        return ''
    if isinstance(df.index,pd.DatetimeIndex):
        if idxfmt == 'Date':
            df.index = df.index.strftime('%m-%d-%y')
        elif idxfmt == 'Datetime':
            df.index = df.index.strftime('%m-%d-%y %H:%M')
            df.index = [re.sub(' 00:00','',x) for x in df.index]
        elif idxfmt == 'Time':
            df.index = df.index.strftime('%h:%M:%S')
        out = df.to_string(index=True)
    else:
        out = df.to_string(index=False)
    for i in range(tabs):
        out = '    '+out.replace('\n','\n    ')
    return out
