import pandas as pd
import datetime as dt
from copy import deepcopy
import re
import sys


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
    spacing = str(max([len(str(x)) for x in dic.keys()])+4)
    for k,v in dic.items():
        if isinstance(v,pd.DataFrame):
            v_str = '\n'+print_dataframe(v,tabs+1)+'\n'
        elif isinstance(v,dict):
            v_str = '\n'+print_dict(v,tabs+1)
        elif isinstance(v,list):
            v_str = [print_dict(x,tabs) for x in v]
            v_str = '\n'+', '.join(v_str)
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


def print_list_table(lis,headers=[]):
    '''Make a string from list of lists with columns for each list

    Parameters
    ----------
    lis : list of lists, data to print
    headers: list of str (optional), headers for each column

    Returns
    -------
    str : string represenation of list of lists as table
    '''
    lis = deepcopy(lis)
    if headers is not None:
        for x,y in zip(lis,headers):
            x.insert(0,'-'*len(y))
            x.insert(0,y)

    # Match lengths
    max_len = max([len(x) for x in lis])
    for x in lis:
        while len(x)<max_len:
            x.append('')
    fmt = '\t'.join(['{%i}' % x for x in range(len(lis))])
    out = []
    for x in zip(*lis):
        out.append(fmt.format(*x))
    out = '\n'.join(out)
    return out


def println(txt):
    '''Print inline without newline
    required due to how ipython doesn't work right with print(..., end='')
    '''
    sys.stdout.write(txt)
    sys.stdout.flush()


def get_next_letter(letter):
    '''gets next letter in the alphabet
    Z -> AA, AZ -> BA, etc
    Preserves case of input, allows mixed-case

    Parameters
    ----------
    letter : str
    '''
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    if letter[-1].islower():
        letters = letters.lower()

    idx = letters.rfind(letter[-1])
    if len(letter) == 1 and idx < (len(letters) - 1):
        return letters[idx+1]
    elif len(letter) == 1:
        return letters[0]*2
    elif len(letter) > 1 and idx < (len(letters) -1):
        return letter[:-1] + letters[idx+1]
    else:
        out = get_next_letter(letter[:-1])
        return out+letters[0]

