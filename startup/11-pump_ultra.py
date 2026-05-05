from ophyd import Device, EpicsMotor, EpicsSignal, EpicsSignalRO
from ophyd import Component as Cpt
import time
import pandas as pd
import numpy as np

class syrng_ultra(Device):
    status = Cpt(EpicsSignalRO, 'STATUS', string=True)
    # 0: Idle, 1: Infusing, 2: Withdrawing, 4: Target Reached
    
    communication = Cpt(EpicsSignal, 'DISABLE', string=True, kind='config')
    # 1: Disabled, 0: Enabled
    
    update_pump = Cpt(EpicsSignal, 'UPDATE.SCAN', string=True, kind='config')
    # 0: Passive, 1: Event, 2: I/O Intr, 3: 10 second 4: 5 second
    # 5: 2 second, 6: 1 second, 7: .5 second, 8: .2 second, 9: .1 second
    
    pump_infuse = Cpt(EpicsSignal, 'IRUN')
    pump_withdraw = Cpt(EpicsSignal, 'WRUN')
    pump_stop = Cpt(EpicsSignal, 'STOP')
    
    
    target_vol = Cpt(EpicsSignal, 'TVOLUME', kind='config')
    target_vol_unit = Cpt(EpicsSignal, 'TVOLUMEUNITS', string=True, kind='config')
    read_target_vol = Cpt(EpicsSignalRO, 'TVOLUME:RBV', kind='config')
    read_target_vol_unit = Cpt(EpicsSignalRO, 'TVOLUMEUNITS:RBV', string=True, kind='config')
    
    clear_infused = Cpt(EpicsSignal, 'CLEARINFUSED')
    clear_withdrawn = Cpt(EpicsSignal, 'CLEARWITHDRAWN')
    
    read_infused = Cpt(EpicsSignalRO, 'IVOLUME:RBV')
    read_withdrawn = Cpt(EpicsSignalRO, 'WVOLUME:RBV')
    read_infused_unit = Cpt(EpicsSignalRO, 'IVOLUMEUNITS:RBV', string=True)
    read_withdrawn_unit = Cpt(EpicsSignalRO, 'WVOLUMEUNITS:RBV', string=True)
    
    infuse_rate = Cpt(EpicsSignal, 'IRATE', kind='hinted')
    infuse_rate_unit = Cpt(EpicsSignal, 'IRATEUNITS', string=True, kind='hinted')
    withdraw_rate = Cpt(EpicsSignal, 'WRATE', kind='hinted')
    withdraw_rate_unit = Cpt(EpicsSignal, 'WRATEUNITS', string=True, kind='hinted')
    
    read_infuse_rate = Cpt(EpicsSignalRO, 'IRATE:RBV', kind='hinted')
    read_infuse_rate_unit = Cpt(EpicsSignalRO, 'IRATEUNITS:RBV', string=True, kind='hinted')
    read_withdraw_rate = Cpt(EpicsSignalRO, 'WRATE:RBV', kind='hinted')
    read_withdraw_rate_unit = Cpt(EpicsSignalRO, 'WRATEUNITS:RBV', string=True, kind='hinted')    
    
    target_time = Cpt(EpicsSignal, 'TTIME', kind='hinted')
    read_target_time = Cpt(EpicsSignalRO, 'TTIME:RBV', kind='hinted')
    
    force = Cpt(EpicsSignal, 'FORCE', kind='hinted')
    read_force = Cpt(EpicsSignalRO, 'FORCE:RBV', kind='hinted')
    
    #pollon = Cpt(EpicsSignal, 'POLLON', kind='config')
    diameter = Cpt(EpicsSignalRO, 'DIAMETER', kind='config')



    
    def _vol_rate_table(self, syringe_material='steel'):
        min_unit = 'nl/min'
        max_unit = 'ml/min'
        
        if syringe_material == 'steel':
            min_vol= {'3.39588nl/min': 2.5, '13.0924nl/min': 8, '52.8105nl/min': 20, 
                    '118.038nl/min': 50, '175.769nl/min': 100}
            _range = {2.5: [3.39588, 3.5265], 8: [13.0924, 13.596], 20: [52.8105, 54.8417], 
                    50: [118.038, 122.578], 100: [175.769, 182.529]}
    
        elif syringe_material == 'glass_H1000': # Hamilton 1000 Series
            min_vol= {'3.06414nl/min': 1, '3.82884nl/min': 1.25, '7.65858nl/min': 2.5, 
                    '15.3096nl/min': 5, '30.6218nl/min': 10, '76.5582nl/min': 25, 
                    '153.111nl/min': 50, '153.111nl/min': 100}
            _range = {1: [3.06414, 3.18204], 1.25: [3.82884, 3.97616], 2.5: [7.65858, 7.95317], 5: [15.3096, 15.8985], 
                    10: [30.6218, 31.7996], 25: [76.5582, 79.5028], 50: [153.111, 159.0], 100: [153.111, 159.0]}
        
        elif syringe_material == 'plastic_BD': # Becton Dickinson
            min_vol= {'3.18636nl/min': 1, '10.6358,nl/min': 3, '20.7422nl/min': 5, 
                    '30.036nl/min': 10, '52.3697nl/min': 20, '67.2661nl/min': 30, 
                    '102.061nl/min': 50, '102.061nl/min': 60}
            _range = {1: [3.18636, 3.30896], 3: [10.6358, 11.0449], 5: [20.7422, 21.5401], 10: [30.036, 31.1913], 
                    20: [52.3697, 54.384], 30: [67.2661, 69.8532], 50: [102.061, 105.986], 60: [102.061, 105.986]}
        
        else:
            raise ValueError(f'Size of {syringe_material} syringe could nor been found.')
        return min_vol, _range    


    
    
    # def find_syringe_type(self, volume, material):
    #     if material == 'steel':
    #         vol_dic = {'2.5': 4.851, '8': 9.525, '20': 19.13, '50': 28.6, '100':34.9}
    #         if self.diameter.get() == vol_dic[str(volume)]:
    #             print(f'Selected Syringe: {volume} mL {material} syringe')
    #         else:
    #             print('Selected syringe does not fit with input. Please check.')
    #     else:
    #         print('You want to use a non-steel syringe. Please check the pump manually.')
            
    
    # def reading_syringe_size(self, input_size):
    #     # Unit of size: ml
    #     dia_vol = {'4.851mm': 2.5, '9.525mm': 8, '19.13mm': 20, '28.6mm': 50, '34.9mm':100}
    #     #diameter = self.diameter.get()
    #     a = (dia_vol[f'{self.diameter.get()}mm'] == input_size)
    #     return a, dia_vol[f'{self.diameter.get()}mm']

       
    def check_pump_condition(self, input_size, syringe_material, wait=False):
        if wait == True:
            time.sleep(1)
            #self.set_infuse_range.put(1, wait=True)
            #self.set_withdraw_range.put(1, wait=True)
        #print('Syringe Diameter: ' + f'{self.diameter.get()} mm' + ' ---> ' + f'{self.steel_syringe_size()} mL {syringe_material} syringe')
        print('\nMake sure select the right syringe size and material on Ultra pump panel.')
        print('There is no way to double check the rate range if the syrnge size/material is not correct.\n')
        # if self.reading_syringe_size(input_size)[0]:
        #     print('Syringe Volume: ' + f'{self.reading_syringe_size(input_size)[1]} mL {syringe_material} syringe')
        # else:
        #     print("(Input size doens't match the reading diameter. Use the input size.)")
        print('Syringe Volume: ' + f'{input_size} mL {syringe_material} syringe')
        _, rate_range = self._vol_rate_table(syringe_material=syringe_material)
        min_rate = rate_range[input_size][0]
        max_rate = rate_range[input_size][1]
        print(f'Min rate: {min_rate} nl/min; Max rate: {max_rate} ul/min')
        print('Pump Stauts: ' + f'{self.status.get()}')
        print('Cmmunication: ' + f'{self.communication.get()} @ {self.update_pump.get()}')
        print('Target Volume: ' + f'{self.read_target_vol.get()} {self.read_target_vol_unit.get()}')
        print('Infuse rate: ' + f'{self.read_infuse_rate.get()} {self.read_infuse_rate_unit.get()}')
        print('Withdraw rate: ' + f'{self.read_withdraw_rate.get()} {self.read_withdraw_rate_unit.get()}')
        print('Infused volume: ' + f'{self.read_infused.get()} {self.read_infused_unit.get()}')
        print('Withdrawn volume: ' + f'{self.read_withdrawn.get()} {self.read_withdrawn_unit.get()}')
            
    
    def initialize_pump(self, clear = True, update = '.5 second'):
        self.communication.put('Enabled')
        self.update_pump.put(update)
        
        if clear == True:
            self.clear_infused.put(1)
            self.clear_withdrawn.put(1)
        time.sleep(1)
        return self.status.get()
    
    def disable_pump(self, clear = True):        
        if clear == True:
            self.clear_infused.put(1)
            self.clear_withdrawn.put(1)
        time.sleep(1)
        self.communication.put('Disabled')
        return self.status.get()
    
    
#     def set_pump(self, clear = False, 
#                  target_vol = 20, target_unit = 'ml', 
#                  infuse_rate = 100, infuse_unit = 'ul/min',
#                  withdraw_rate = 100, withdraw_unit = 'ul/min'):
        
#         if clear == True:
#             self.clear_infused.put(1)
#             self.clear_withdrawn.put(1)
        
#         c = vol_unit_converter(v0=target_unit, v1='ml')
#         if target_vol*c > self.steel_syringe_size():
#             raise ValueError (f'Input target volume {target_vol*c} mL larger than syringe size.')        
#         self.target_vol_unit.put(target_unit, wait=True)
#         self.target_vol.put(target_vol, wait=True)
        
#         min_unit = self.show_steel_max_min_rate()[1]
#         max_unit = self.show_steel_max_min_rate()[3]
        
#         const1_max = vol_unit_converter(v0=infuse_unit[:2], v1=max_unit[:2])/t_unit_converter(t0=infuse_unit[3:], t1=max_unit[3:])
#         const1_min = vol_unit_converter(v0=infuse_unit[:2], v1=min_unit[:2])/t_unit_converter(t0=infuse_unit[3:], t1=min_unit[3:])
        
#         if infuse_rate*const1_max > self.show_steel_max_min_rate()[2]:
#             raise ValueError(f'Input infuse rate {infuse_rate*const1_max:.3f} {max_unit} larger than allowed value.')
#         elif infuse_rate*const1_min < self.show_steel_max_min_rate()[0]:
#             raise ValueError(f'Input infuse rate {infuse_rate*const1_min:.3f} {min_unit} smaller than allowed value.')
#         else:
#             self.infuse_rate_unit.put(infuse_unit, wait=True)
#             self.infuse_rate.put(infuse_rate, wait=True)
              
#         const2_max = vol_unit_converter(v0=withdraw_unit[:2], v1=max_unit[:2])/t_unit_converter(t0=withdraw_unit[3:], t1=max_unit[3:])
#         const2_min = vol_unit_converter(v0=withdraw_unit[:2], v1=min_unit[:2])/t_unit_converter(t0=withdraw_unit[3:], t1=min_unit[3:])
#         if withdraw_rate*const2_max > self.show_steel_max_min_rate()[2]:
#             raise ValueError(f'Input withdraw rate {withdraw_rate*const2_max:.3f} {max_unit} larger than allowed value.')
#         elif withdraw_rate*const2_min < self.show_steel_max_min_rate()[0]:
#             raise ValueError(f'Input withdraw rate {withdraw_rate*const2_min:.3f} {min_unit} smaller than allowed value.')
#         else:
#             self.withdraw_rate_unit.put(withdraw_unit, wait=True)
#             self.withdraw_rate.put(withdraw_rate, wait=True)


    # def set_infuse(self, input_size, clear = False, 
    #                target_vol = 20, target_unit = 'ml', 
    #                infuse_rate = 100, infuse_unit = 'ul/min'):
    #     if clear == True:
    #         yield from bps.abs_set(self.clear_infused, 1, wait=True)
    #         yield from bps.abs_set(self.clear_withdrawn, 1, wait=True)
        
    #     if self.reading_syringe_size(input_size)[0]:
    #         size = self.reading_syringe_size(input_size)[1]
    #     else:
    #         size = input_size
        
    #     c = vol_unit_converter(v0=target_unit, v1='ml')
    #     if target_vol*c > size:
    #         raise ValueError (f'Input target volume {target_vol*c} mL larger than syringe size.')        
    #     yield from bps.abs_set(self.target_vol_unit, target_unit, wait=True)
    #     yield from bps.abs_set(self.target_vol, target_vol, wait=True)
        
    #     min_unit = self.show_steel_max_min_rate(input_size)[1]
    #     max_unit = self.show_steel_max_min_rate(input_size)[3]

    #     const1_max = vol_unit_converter(v0=infuse_unit[:2], v1=max_unit[:2])/t_unit_converter(t0=infuse_unit[3:], t1=max_unit[3:])
    #     const1_min = vol_unit_converter(v0=infuse_unit[:2], v1=min_unit[:2])/t_unit_converter(t0=infuse_unit[3:], t1=min_unit[3:])
        
    #     if infuse_rate*const1_max > self.show_steel_max_min_rate(input_size)[2]:
    #         raise ValueError(f'Input infuse rate {infuse_rate*const1_max:.3f} {max_unit} larger than allowed value.')
    #     elif infuse_rate*const1_min < self.show_steel_max_min_rate(input_size)[0]:
    #         raise ValueError(f'Input infuse rate {infuse_rate*const1_min:.3f} {min_unit} smaller than allowed value.')
    #     else:
    #         yield from bps.abs_set(self.infuse_rate_unit, infuse_unit, wait=True)
    #         yield from bps.abs_set(self.infuse_rate, infuse_rate, wait=True)



    def set_infuse2(self, input_size, syringe_material='steel', clear = False, 
                   set_target = True, target_vol = 20, target_unit = 'ml', 
                   infuse_rate = 100, infuse_unit = 'ul/min'):
        if clear == True:
            yield from bps.mv(self.clear_infused, 1, self.clear_withdrawn, 1)
            # yield from bps.abs_set(self.clear_infused, 1, wait=True)
            # yield from bps.abs_set(self.clear_withdrawn, 1, wait=True)
        
        if self.communication.get() == 'Disabled':
            yield from bps.mv(self.communication, 0, self.update_pump, 6)

        def _syringe_size(input_size, syringe_material):
            #yield from bps.mv(self.set_infuse_range, 1, self.set_withdraw_range, 1)
            min_vol = self._vol_rate_table(syringe_material=syringe_material)[0]
            a = (min_vol[f'{self.read_min_infuse.get()}nl/min'] == input_size)
            return a, min_vol[f'{self.read_min_infuse.get()}nl/min']
        
        # if _syringe_size(input_size, syringe_material)[0]:
        #     size = _syringe_size(input_size, syringe_material)[1]
        # else:
        #     #size = input_size
        #     raise ValueError('Input size could not been found. Check selected syringe type & size.')
        
        size = input_size
        
        if set_target:
            c = vol_unit_converter(v0=target_unit, v1='ml')
            if target_vol*c > size:
                raise ValueError (f'Input target volume {target_vol*c} mL larger than syringe size.')
            yield from bps.abs_set(self.target_vol_unit, target_unit, wait=True)
            yield from bps.abs_set(self.target_vol, target_vol, wait=True)
        
        min_unit = 'nl/min'
        max_unit = 'ml/min'
        min_in_device = self._vol_rate_table(syringe_material=syringe_material)[1][size][0]
        max_in_device = self._vol_rate_table(syringe_material=syringe_material)[1][size][1]

        # const1_max = vol_unit_converter(v0=infuse_unit[:2], v1=max_unit[:2])/t_unit_converter(t0=infuse_unit[3:], t1=max_unit[3:])
        # const1_min = vol_unit_converter(v0=infuse_unit[:2], v1=min_unit[:2])/t_unit_converter(t0=infuse_unit[3:], t1=min_unit[3:])      
        const1_max = rate_unit_converter(r0=infuse_unit, r1=max_unit)
        const1_min = rate_unit_converter(r0=infuse_unit, r1=min_unit)
        
        if infuse_rate*const1_max > max_in_device:
            raise ValueError(f'Input infuse rate {infuse_rate*const1_max:.3f} {max_unit} larger than allowed value.')
        elif infuse_rate*const1_min < min_in_device:
            raise ValueError(f'Input infuse rate {infuse_rate*const1_min:.3f} {min_unit} smaller than allowed value.')
        
        yield from bps.abs_set(self.infuse_rate_unit, infuse_unit, wait=True)
        yield from bps.abs_set(self.infuse_rate, infuse_rate, wait=True)
        yield from bps.sleep(1)



    def set_withdraw(self, input_size, syringe_material='steel', clear = False, 
                     target_vol = 20, target_unit = 'ml', 
                     withdraw_rate = 100, withdraw_unit = 'ul/min'):
        if clear == True:
            yield from bps.mv(self.clear_infused, 1, self.clear_withdrawn, 1)
            # yield from bps.abs_set(self.clear_infused, 1, wait=True)
            # yield from bps.abs_set(self.clear_withdrawn, 1, wait=True)
        
        yield from bps.mv(self.set_infuse_range, 1, self.set_withdraw_range, 1)

        def _syringe_size(input_size, syringe_material):
            #yield from bps.mv(self.set_infuse_range, 1, self.set_withdraw_range, 1)
            min_vol = set._vol_rate_table(syringe_material=syringe_material)[0]
            a = (min_vol[f'{self.read_min_infuse.get()}nl/min'] == input_size)
            return a, min_vol[f'{self.read_min_infuse.get()}nl/min']
        
        if _syringe_size(input_size, syringe_material)[0]:
            size = _syringe_size(input_size, syringe_material)[1]
        else:
            #size = input_size
            raise ValueError('Input size could not been found. Check selected syringe type & size.')
        
        c = vol_unit_converter(v0=target_unit, v1='ml')
        
        min_unit = 'nl/min'
        max_unit = 'ml/min'
        min_in_theory = set._vol_rate_table(syringe_material=syringe_material)[1][size][0]
        max_in_theory = set._vol_rate_table(syringe_material=syringe_material)[1][size][1]

        const1_max = vol_unit_converter(v0=withdraw_unit[:2], v1=max_unit[:2])/t_unit_converter(t0=withdraw_unit[3:], t1=max_unit[3:])
        const1_min = vol_unit_converter(v0=withdraw_unit[:2], v1=min_unit[:2])/t_unit_converter(t0=withdraw_unit[3:], t1=min_unit[3:])
        
        if target_vol*c > size:
            raise ValueError (f'Input target volume {target_vol*c} mL larger than syringe size.')        
        elif withdraw_rate*const1_max > max_in_theory:
            raise ValueError(f'Input withdraw rate {withdraw_rate*const1_max:.3f} {max_unit} larger than allowed value.')
        elif withdraw_rate*const1_min < min_in_theory:
            raise ValueError(f'Input withdraw rate {withdraw_rate*const1_min:.3f} {min_unit} smaller than allowed value.')
        else:
            yield from bps.mv(self.target_vol_unit, target_unit, 
                              self.target_vol, target_vol, 
                              self.withdraw_rate_unit, withdraw_unit, 
                              self.withdraw_rate, withdraw_rate)
            # yield from bps.abs_set(self.infuse_rate_unit, infuse_unit, wait=True)
            # yield from bps.abs_set(self.infuse_rate, infuse_rate, wait=True)



    def infuse_pump(self, clear = False):
        if clear == True:
            self.clear_infused.put(1)
            self.clear_withdrawn.put(1)
        self.pump_infuse.put(1)
        time.sleep(1)
        return self.status.get()
    
    def infuse_pump2(self, clear = False):
        if clear == True:
            yield from bps.abs_set(self.clear_infused, 1, wait=True)
            yield from bps.abs_set(self.clear_withdrawn, 1, wait=True)
        yield from bps.abs_set(self.pump_infuse, 1, wait=True)
        yield from bps.sleep(1)
        
        #time.sleep(1)  
        ## There is a specific way in RE{}; time.sleep is dangerous for RE{} --> use bps.sleep()
        #return self.status.get()
               
        
    def withdraw_pump(self, clear = False):
        if clear == True:
            self.clear_infused.put(1)
            self.clear_withdrawn.put(1)
        self.pump_withdraw.put(1)
        time.sleep(1)
        return self.status.get()
    
    def withdraw_pump2(self, clear = False):
        if clear == True:
            yield from bps.abs_set(self.clear_infused, 1, wait=True)
            yield from bps.abs_set(self.clear_withdrawn, 1, wait=True)
        yield from bps.abs_set(self.pump_withdraw, 1, wait=True)
        yield from bps.sleep(1)

        
    def stop_pump(self, clear = False):
        self.pump_stop.put(1)
        if clear == True:
            self.clear_infused.put(1)
            self.clear_withdrawn.put(1)
        time.sleep(1)
        return self.status.get()
    
    def stop_pump2(self, clear = False):
        yield from bps.abs_set(self.pump_stop, 1, wait=True)
        if clear == True:
            yield from bps.abs_set(self.clear_infused, 1, wait=True)
            yield from bps.abs_set(self.clear_withdrawn, 1, wait=True)
        yield from bps.sleep(1)

    
    def show_steel_max_min_rate(self, input_size):
        min_unit = 'nl/min'
        max_unit = 'ml/min'
        
        if self.reading_syringe_size(input_size)[0]:
            size = self.reading_syringe_size(input_size)[1]
        else:
            size = input_size
            
        if size == 2.5:
            min_rate = 3.39588
            max_rate = 3.5265
        elif size == 8:
            min_rate = 13.0924
            max_rate = 13.596
        elif size == 20:
            min_rate = 52.8105
            max_rate = 54.8417
        elif size == 50:
            min_rate = 118.038
            max_rate = 122.578
        elif size == 100:
            min_rate = 175.769
            max_rate = 182.529 
        else:
            min_rate = 'Unkonwn'
            max_rate = 'Unknown'
        return min_rate, min_unit, max_rate, max_unit
    
    def stage(self):
        print('I am staging.')
        super().stage()

    def unstage(self):
        print('I am unstaging.')
        super().unstage()
    




def _ultra_vol_rate_table(syringe_material='steel'):
    min_unit = 'nl/min'
    max_unit = 'ml/min'
    
    if syringe_material == 'steel':
        min_vol= {'3.39588nl/min': 2.5, '13.0924nl/min': 8, '52.8105nl/min': 20, 
                  '118.038nl/min': 50, '175.769nl/min': 100}
        _range = {2.5: [3.39588, 3.5265], 8: [13.0924, 13.596], 20: [52.8105, 54.8417], 
                  50: [118.038, 122.578], 100: [175.769, 182.529]}
   
    elif syringe_material == 'glass_H1000': # Hamilton 1000 Series
        min_vol= {'3.06414nl/min': 1, '3.82884nl/min': 1.25, '7.65858nl/min': 2.5, 
                  '15.3096nl/min': 5, '30.6218nl/min': 10, '76.5582nl/min': 25, 
                  '153.111nl/min': 50, '153.111nl/min': 100}
        _range = {1: [3.06414, 3.18204], 1.25: [3.82884, 3.97616], 2.5: [7.65858, 7.95317], 5: [15.3096, 15.8985], 
                  10: [30.6218, 31.7996], 25: [76.5582, 79.5028], 50: [153.111, 159.0], 100: [153.111, 159.0]}
    
    elif syringe_material == 'plastic_BD': # Becton Dickinson
        min_vol= {'3.18636nl/min': 1, '10.6358,nl/min': 3, '20.7422nl/min': 5, 
                  '30.036nl/min': 10, '52.3697nl/min': 20, '67.2661nl/min': 30, 
                  '102.061nl/min': 50, '102.061nl/min': 60}
        _range = {1: [3.18636, 3.30896], 3: [10.6358, 11.0449], 5: [20.7422, 21.5401], 10: [30.036, 31.1913], 
                  20: [52.3697, 54.384], 30: [67.2661, 69.8532], 50: [102.061, 105.986], 60: [102.061, 105.986]}
    
    else:
        raise ValueError(f'Size of {syringe_material} syringe could nor been found.')
    return min_vol, _range



    
def vol_unit_converter(v0 = 'ul', v1 = 'ml'):
    vol_unit = ['pl', 'nl', 'ul', 'ml']
    vol_frame = pd.DataFrame(data={'pl': np.geomspace(1, 1E9, num=4), 'nl': np.geomspace(1E-3, 1E6, num=4),
                                   'ul': np.geomspace(1E-6, 1E3, num=4), 'ml': np.geomspace(1E-9, 1, num=4)}, index=vol_unit)
    return vol_frame.loc[v0, v1]


def t_unit_converter(t0 = 'min', t1 = 'min'):
    t_unit = ['sec', 'min', 'hr']
    t_frame = pd.DataFrame(data={'sec': np.geomspace(1, 3600, num=3), 
                                 'min': np.geomspace(1/60, 60, num=3), 
                                 'hr' : np.geomspace(1/3600, 1, num=3)}, index=t_unit)
    return t_frame.loc[t0, t1]


def syringe_diameter(volume, material='steel'):
    if material == 'steel':
        vol_dic = {'2.5': 4.851, '8': 9.525, '20': 19.13, '50': 28.6, '100':34.9}
        if str(volume) in vol_dic:
            print(f'{volume} mL {material} Syringe Diameter is {vol_dic[str(volume)]} mm.')
        else: raise ValueError('Input volume is not supported by Harvard stainless steel syringe.')
    else:
        print('Please check the diameter of non-steel syringe manually.')



# ultra1 = syrng_ultra('XF:28IDC-ES:1{Pump:Syrng-Ultra:1}:', name='Pump_Ultra1', 
#                      read_attrs=['status', 'communication', 'update_pump', 
#                                  'read_target_vol', 'read_target_vol_unit',
#                                  'read_infused', 'read_infused_unit', 
#                                  'read_withdrawn', 'read_withdrawn_unit',
#                                  'read_infuse_rate', 'read_infuse_rate_unit',
#                                  'read_withdraw_rate', 'read_withdraw_rate_unit'])

# ultra2 = syrng_ultra('XF:28IDC-ES:1{Pump:Syrng-Ultra:2}:', name='Pump_Ultra2', 
#                      read_attrs=['status', 'communication', 'update_pump', 
#                                  'read_target_vol', 'read_target_vol_unit',
#                                  'read_infused', 'read_infused_unit', 
#                                  'read_withdrawn', 'read_withdrawn_unit',
#                                  'read_infuse_rate', 'read_infuse_rate_unit',
#                                  'read_withdraw_rate', 'read_withdraw_rate_unit'])