from ophyd import Device, EpicsMotor, EpicsSignal, EpicsSignalRO
from ophyd import Component as Cpt
#import time
#import pandas as pd
#import numpy as np


# LED = EpicsSignal('XF:28IDC-ES:1{Light:Flu-LED:1}Cmd', name='LED_M365LP1', string=True, kind='hinted')
# # 0: 'Low'; 1: 'High'

# deuterium = EpicsSignal('XF:28IDC-ES:1{Light:Abs-Dut:1}Cmd', name='Deuterium', string=True, kind='Config')
# # 0: 'Low'; 1: 'High'

# halogen = EpicsSignal('XF:28IDC-ES:1{Light:Abs-Hal:1}Cmd', name='Halogen', string=True, kind='Config')
# # 0: 'Low'; 1: 'High'

# UV_shutter = EpicsSignal('XF:28IDC-ES:1{Light:Abs-Sht:1}Cmd', name='UV_shutter', string=True, kind='hinted')
# # 0: 'Low' --> shutter close.; 1: 'High' --> Shutter open.


def LED_on():
    yield from bps.abs_set(LED, 'High', wait=True)
    print(f'LED light is {LED.get()}.')
    # return 'test'
    #st=DeviceStatus()
    #LED.put('High', wait=True)
    #return (yield from bps.rd(LED))
    
# def LED_on_2():
#     #yield from bps.abs_set(LED, 'High', wait=True)
#     LED.put('High', wait=True)
#     return LED.get()

# def LED_on_3():
#     #yield from bps.abs_set(LED, 'High', wait=True)
#     #LED.put('High', wait=True)
#     return (yield from bps.trigger_and_read(LED))
    
def LED_off():
    yield from bps.abs_set(LED, 'Low', wait=True)
    print(f'LED light is {LED.get()}')
    #LED.put('Low', wait=True)
    # return LED.get()
    
# def LED_off_2():
#     #yield from bps.abs_set(LED, 'Low', wait=True)
#     LED.put('Low', wait=True)
#     return LED.get()
    
def shutter_open():
    yield from bps.abs_set(UV_shutter, 'High', wait=True)
    print(f'UV shutter is {UV_shutter.get()}')
    #UV_shutter.put('High', wait=True)
    #return UV_shutter.get()
    #return (yield from bps.rd(UV_shutter))
    
def shutter_close():
    yield from bps.abs_set(UV_shutter, 'Low', wait=True)
    print(f'UV shutter is {UV_shutter.get()}')
    #UV_shutter.put('Low', wait=True)
    #return UV_shutter.get()
    
def deuterium_on():
    yield from bps.abs_set(deuterium, 'High', wait=True)
    print(f'Deuterium light is {deuterium.get()}')
    
def deuterium_off():
    yield from bps.abs_set(deuterium, 'Low', wait=True)
    print(f'Deuterium light is {deuterium.get()}')
    
def halogen_on():
    yield from bps.abs_set(halogen, 'High', wait=True)
    print(f'Halogen light is {halogen.get()}')
    
def halogen_off():
    yield from bps.abs_set(halogen, 'Low', wait=True)
    print(f'Halogen light is {halogen.get()}')