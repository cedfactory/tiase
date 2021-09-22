import pandas as pd
import numpy as np

def get_constant(amplitude, length):
    return amplitude * np.ones(length)

def get_linear(a, b, length):
    x = np.arange(length)
    y = a*x + b
    return y

def get_sinusoid(length=100, amplitude=1, frequency=.1, phi=0, height = 0):
    x = np.arange(length)
    y = amplitude*np.sin(frequency*x+phi)+height
    return y

def add_noise(y, amplitude):
    return np.array(y+amplitude*np.random.randn(len(y)))

def create_dataframe(y, mult=1):    
    op = add_noise(y, mult)
    cl = add_noise(y, mult)
    hi = np.maximum(op, cl) + np.array(mult*np.abs(np.random.randn(len(y))))
    lo = np.minimum(op, cl) - np.array(mult*np.abs(np.random.randn(len(y))))
    vol = np.array(100 + mult*np.abs(np.random.randn(len(y))))
    dti = pd.date_range('2000-01-01', periods=len(y), freq='D')
    idx = np.arange(len(y))

    data = {'Date': dti ,
        'Open': op, 
        'Close': cl, 
        'High': hi, 
        'Low': lo,
        'Volume': vol
        } 

    df = pd.DataFrame(data, index=idx)
    df.set_index('Date',inplace=True)
    return df
