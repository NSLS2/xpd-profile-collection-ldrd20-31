from gc import collect
import logging
import time

import epics

from ophyd import (Device, Component as Cpt, FormattedComponent as FC,
                   Signal)
from ophyd import (EpicsSignal, EpicsSignalRO, DeviceStatus, DerivedSignal)
from ophyd.areadetector import EpicsSignalWithRBV as SignalWithRBV
from ophyd.status import SubscriptionStatus

class QEProTEC(Device):

    # Thermal electric cooler settings
    tec = Cpt(SignalWithRBV, 'TEC')
    tec_temp = Cpt(SignalWithRBV, 'TEC_TEMP', kind='config')
    curr_tec_temp = Cpt(EpicsSignalRO, 'CURR_TEC_TEMP_RBV')

    def __init__(self, *args, tolerance=1, **kwargs):
        self.tolerance = tolerance
        super().__init__(*args, **kwargs)

    def set(self, value):

        def check_setpoint(value, old_value, **kwargs):
            if abs(value - self.tec_temp.get()) < self.tolerance:
                print(f'Reached setpoint {self.tec_temp.get()}.')
                return True
            return False

        status = SubscriptionStatus(self.curr_tec_temp, run=False, callback=check_setpoint)
        self.tec_temp.put(value)
        self.tec.put(1)

        return status


class QEPro(Device):

    # Device information
    serial = Cpt(EpicsSignal, 'SERIAL')
    model = Cpt(EpicsSignal, 'MODEL')

    # Device Status
    status = Cpt(EpicsSignal, 'STATUS')
    status_msg = Cpt(EpicsSignal, 'STATUS_MSG', string=True)
    device_connected = Cpt(EpicsSignalRO, 'CONNECTED_RBV')

    # Utility signal that periodically checks device temps.
    __check_status = Cpt(EpicsSignal, 'CHECK_STATUS')

    # Bit array outlining which features are supported by the device
    features = Cpt(EpicsSignalRO, 'FEATURES_RBV')
    
    # Togglable features (if supported)
    strobe = Cpt(SignalWithRBV, 'STROBE')
    electric_dark_correction = Cpt(SignalWithRBV, 'EDC', kind='normal')
    non_linearity_correction = Cpt(SignalWithRBV, 'NLC', kind='normal')
    shutter = Cpt(SignalWithRBV, 'SHUTTER')
    
    # Thermal electric cooler
    tec_device = Cpt(QEProTEC, '')
    
    # Light source feature signals
    light_source = Cpt(SignalWithRBV, 'LIGHT_SOURCE')
    light_source_intensity = Cpt(SignalWithRBV, 'LIGHT_SOURCE_INTENSITY')
    light_source_count = Cpt(EpicsSignalRO, 'LIGHT_SOURCE_COUNT_RBV')

    # Signals for specifying the number of spectra to average and counter for spectra
    # collected in current scan
    num_spectra = Cpt(SignalWithRBV, 'NUM_SPECTRA', kind='normal')
    spectra_collected = Cpt(EpicsSignalRO, 'SPECTRA_COLLECTED_RBV')

    # Integration time settings (in ms)
    int_min_time = Cpt(EpicsSignalRO, 'INT_MIN_TIME_RBV')
    int_max_time = Cpt(EpicsSignalRO, 'INT_MAX_TIME_RBV')
    integration_time = Cpt(SignalWithRBV, 'INTEGRATION_TIME', kind='normal')
    
    # Internal buffer feature settings
    buff_min_capacity = Cpt(EpicsSignalRO, 'BUFF_MIN_CAPACITY_RBV')
    buff_max_capacity = Cpt(EpicsSignalRO, 'BUFF_MAX_CAPACITY_RBV')
    buff_capacity = Cpt(SignalWithRBV, 'BUFF_CAPACITY')
    buff_element_count = Cpt(EpicsSignalRO, 'BUFF_ELEMENT_COUNT_RBV')

    # Formatted Spectra
    output = Cpt(EpicsSignal, 'OUTPUT', kind='normal')
    sample = Cpt(EpicsSignal, 'SAMPLE', kind='normal')
    dark = Cpt(EpicsSignal, 'DARK', kind='normal')
    reference = Cpt(EpicsSignal, 'REFERENCE', kind='normal')
    
    # Length of spectrum (in pixels)
    formatted_spectrum_len = Cpt(EpicsSignalRO, 'FORMATTED_SPECTRUM_LEN_RBV')

    # X-axis format and array
    x_axis = Cpt(EpicsSignal, 'X_AXIS', kind='normal')
    x_axis_format = Cpt(SignalWithRBV, 'X_AXIS_FORMAT')

    # Dark/Ref available signals
    dark_available = Cpt(EpicsSignalRO, 'DARK_AVAILABLE_RBV')
    ref_available = Cpt(EpicsSignalRO, 'REF_AVAILABLE_RBV')

    # Collection settings and start signals.
    acquire = Cpt(SignalWithRBV, 'COLLECT')
    collect_mode = Cpt(SignalWithRBV, 'COLLECT_MODE', kind='normal')
    spectrum_type = Cpt(SignalWithRBV, 'SPECTRUM_TYPE', kind='normal')
    correction = Cpt(SignalWithRBV, 'CORRECTION', kind='normal')
    trigger_mode = Cpt(SignalWithRBV, 'TRIGGER_MODE')


    @property
    def has_nlc_feature(self):
        return self.features.get() & 32

    @property
    def has_lightsource_feature(self):
        return self.features.get() & 16

    @property
    def has_edc_feature(self):
        return self.features.get() & 8

    @property
    def has_buffer_feature(self):
        return self.features.get() & 4

    @property
    def has_tec_feature(self):
        return self.features.get() & 2

    @property
    def has_irrad_feature(self):
        return self.features.get() & 1


    def describe(self):
        description = super().describe()
        # description[f"{self.name}_status_msg"].setdefault("dtype_str", "<i8")
        # description[f"{self.name}_output"].setdefault("dtype_str", "|O")
        return description
    
    
    def set_temp(self, temperature):
        self.tec_device.set(temperature).wait()


    def get_dark_frame(self):

        current_spectrum = self.spectrum_type.get()
        self.spectrum_type.put('Dark')
        self.acquire.put(1, wait=True)
        time.sleep(1)
        self.spectrum_type.put(current_spectrum)
    
    def get_dark_frame2(self):

        current_spectrum = self.spectrum_type.get()
        yield from bps.abs_set(self.spectrum_type, 'Dark', wait=True)
        yield from bps.abs_set(self.acquire, 1, wait=True)
        yield from bps.sleep(1)
        yield from bps.abs_set(self.spectrum_type, current_spectrum, wait=True)
        
    
    def get_reference_frame(self):

        current_spectrum = self.spectrum_type.get()
        self.spectrum_type.put('Reference')
        self.acquire.put(1, wait=True)
        time.sleep(1)
        self.spectrum_type.put(current_spectrum)
        
        
    def get_reference_frame2(self):

        current_spectrum = self.spectrum_type.get()
        yield from bps.abs_set(self.spectrum_type, 'Reference', wait=True)
        yield from bps.abs_set(self.acquire, 1, wait=True)
        yield from bps.sleep(1)
        yield from bps.abs_set(self.spectrum_type, current_spectrum, wait=True)
        
    
    def take_ref_bkg(self, integration_time=15, num_spectra_to_average=16, buffer=3, electric_dark_correction=True):
        yield from self.setup_collection2(integration_time=integration_time, num_spectra_to_average=num_spectra_to_average, buffer=buffer, 
                                         spectrum_type='Absorbtion', correction_type='Reference', 
                                         electric_dark_correction=True)
        # yield from LED_off()
        # yield from shutter_close()
        yield from bps.mv(LED, 'Low', UV_shutter, 'Low')
        yield from self.get_dark_frame2()
        yield from bps.mv(UV_shutter, 'High')
        yield from self.get_reference_frame2()
        
        
    def take_ref_bkg2(self, integration_time=15, num_spectra_to_average=16, buffer=3, electric_dark_correction=True):
        yield from self.setup_collection2(integration_time=integration_time, num_spectra_to_average=num_spectra_to_average, buffer=buffer, 
                                         spectrum_type='Absorbtion', correction_type='Reference', 
                                         electric_dark_correction=True)
        # yield from LED_off()
        # yield from shutter_close()
        yield from bps.mv(LED, 'Low', UV_shutter, 'Low')
        yield from bps.sleep(5)
        yield from self.get_dark_frame2()
        yield from count([self])
        yield from bps.mv(UV_shutter, 'High')
        yield from bps.sleep(5)
        yield from self.get_reference_frame2()
        yield from count([self])

    
    def take_ref_bkg3(self, integration_time=15, num_spectra_to_average=16, 
                      buffer=3, electric_dark_correction=True, ref_name='test', csv_path=None):
        yield from self.setup_collection2(integration_time=integration_time, num_spectra_to_average=num_spectra_to_average, buffer=buffer, 
                                         spectrum_type='Absorbtion', correction_type='Reference', 
                                         electric_dark_correction=True)
        # yield from LED_off()
        # yield from shutter_close()
        yield from bps.mv(LED, 'Low', UV_shutter, 'Low')
        yield from bps.sleep(5)
        yield from self.get_dark_frame2()
        uid = (yield from count([self], md = {'note':'Dark'}))
        if csv_path != None:
            print(f'Export dark file to {csv_path}...')
            self.export_from_scan(uid, csv_path, sample_type=f'Dark_{integration_time}ms')

        yield from bps.mv(UV_shutter, 'High')
        yield from bps.sleep(2)
        yield from self.get_reference_frame2()
        uid = (yield from count([self], md = {'note':ref_name}))
        
        yield from bps.mv(LED, 'Low', UV_shutter, 'Low')

        if csv_path != None:
            print(f'Export reference file to {csv_path}...')
            self.export_from_scan(uid, csv_path, sample_type=f'{ref_name}_{integration_time}ms')


    def setup_collection(self, integration_time=100, num_spectra_to_average=10, buffer=3,
                         spectrum_type='Absorbtion', correction_type='Reference', 
                         electric_dark_correction=True):
        
        # For absorbance: spectrum_type='Absorbtion', correction_type='Reference'
        # For fluorescence: spectrum_type='Corrected Sample', correction_type='Dark'
        
        self.integration_time.put(integration_time)
        self.num_spectra.put(num_spectra_to_average)
        self.buff_capacity.put(buffer)
        if num_spectra_to_average > 1:
            self.collect_mode.put('Average')
        else:
            self.collect_mode.put('Single')

        if electric_dark_correction:
            self.electric_dark_correction.put(1)

        self.correction.put(correction_type)

        self.spectrum_type.put(spectrum_type)
        
        
    def setup_collection2(self, integration_time=100, num_spectra_to_average=10, buffer=3,
                         spectrum_type='Absorbtion', correction_type='Reference', 
                         electric_dark_correction=True):
        
        # For absorbance: spectrum_type='Absorbtion', correction_type='Reference'
        # For fluorescence: spectrum_type='Corrected Sample', correction_type='Dark'
        
        yield from bps.abs_set(self.integration_time, integration_time, wait=True)
        yield from bps.abs_set(self.num_spectra, num_spectra_to_average, wait=True)
        yield from bps.abs_set(self.buff_capacity, buffer, wait=True)
        if num_spectra_to_average > 1:
            yield from bps.abs_set(self.collect_mode, 'Average', wait=True)
        else:
            yield from bps.abs_set(self.collect_mode, 'Single', wait=True)

        if electric_dark_correction:
            yield from bps.abs_set(self.electric_dark_correction, 1, wait=True)

        yield from bps.abs_set(self.correction, correction_type, wait=True)
        yield from bps.abs_set(self.spectrum_type, spectrum_type, wait=True)


    def grab_frame(self):

        def is_done(value, old_value, **kwargs):
            if old_value == 1 and value ==0:
                return True
            return False

        status = SubscriptionStatus(self.acquire, run=False, callback=is_done)

        self.acquire.put(1)
        return status

    
    def grab_frame2(self):

        def is_done(value, old_value, **kwargs):
            if old_value == 1 and value ==0:
                return True
            return False

        status = SubscriptionStatus(self.acquire, run=False, callback=is_done)

        yield from bps.abs_set(self.acquire, 1, wait=True)
        return status
    
    
    def trigger(self):
        #self.grab_frame().wait()
        # return self.grab_frame()
        # return (yield from self.grab_frame2())
        return self.grab_frame()

    def write_as_csv(self, write_path):
    
        print(f'Writing out CSV file to {write_path}...')

        with open(write_path, 'w') as fp:
            x_axis_data = self.x_axis.get()
            output_data = self.output.get()
            sample_data = self.sample.get()
            dark_data = self.dark.get()
            reference_data = self.reference.get()
            if self.spectrum_type.get(as_string=True) == 'Absorbtion':
                fp.write('Energy,Dark,Reference,Sample,Absorbtion\n')
            else:
                fp.write('Energy,Dark,Raw Sample,Corrected Sample\n')

            for i in range(len(output_data)):
                if self.spectrum_type.get(as_string=True) == 'Absorbtion':
                    fp.write(f'{x_axis_data[i]},{dark_data[i]},{reference_data[i]},{sample_data[i]},{output_data[i]}\n')
                else:
                    fp.write(f'{x_axis_data[i]},{dark_data[i]},{sample_data[i]},{output_data[i]}\n')

            print('Done.')

    def plot_spectra(self):
        x_axis_data = self.x_axis.get()
        output_data = self.output.get()

        x_axis_label = self.x_axis_format.get(as_string=True)
        y_axis_label = self.spectrum_type.get(as_string=True)


        plt.plot(x_axis_data, output_data)
        plt.xlabel(x_axis_label)
        plt.ylabel(y_axis_label)
        plt.show()
        


    def take_uvvis_save_csv(self, sample_type='test', plot=False, csv_path=None, data_agent='db', 
                            spectrum_type='Absorbtion', correction_type='Reference', 
                            pump_list=None, precursor_list=None, mixer=None, note=None, md=None):
        
        if (pump_list != None and precursor_list != None):
            _md = {"pumps" : [pump.name for pump in pump_list], 
                    "precursors" : precursor_list, 
                    "infuse_rate" : [pump.read_infuse_rate.get() for pump in pump_list], 
                    "infuse_rate_unit" : [pump.read_infuse_rate_unit.get() for pump in pump_list],
                    "pump_status" : [pump.status.get() for pump in pump_list], 
                    "uvvis" :[spectrum_type, correction_type, self.integration_time.get(), self.num_spectra.get(), self.buff_capacity.get()], 
                    "mixer": mixer,
                    "sample_type": sample_type,
                    "note" : note if note else "None"}
            _md.update(md or {})
        
        if (pump_list == None and precursor_list == None):
            _md = { "uvvis" :[spectrum_type, correction_type, self.integration_time.get(), self.num_spectra.get(), self.buff_capacity.get()], 
                    "mixer": ['exsitu measurement'],
                    "sample_type": sample_type,
                    "note" : note if note else "None"}
            _md.update(md or {})
        
        # For absorbance: spectrum_type='Absorbtion', correction_type='Reference'
        # For fluorescence: spectrum_type='Corrected Sample', correction_type='Dark'
        
        # self.correction.put(correction_type)
        # self.spectrum_type.put(spectrum_type)
        
        if spectrum_type == 'Absorbtion':
            if LED.get()=='Low' and UV_shutter.get()=='High' and self.correction.get()==correction_type and self.spectrum_type.get()==spectrum_type:
                uid = (yield from count([self], md=_md))
            else:
                yield from bps.abs_set(self.correction, correction_type, wait=True)
                yield from bps.abs_set(self.spectrum_type, spectrum_type, wait=True)
                # yield from LED_off()
                # yield from shutter_open()
                yield from bps.mv(LED, 'Low', UV_shutter, 'High')
                yield from bps.sleep(2)
                uid = (yield from count([self], md=_md))
            
            
        else:
            if LED.get()=='High' and UV_shutter.get()=='Low' and self.correction.get()==correction_type and self.spectrum_type.get()==spectrum_type:
                uid = (yield from count([self], md=_md))
            else:
                yield from bps.abs_set(self.correction, correction_type, wait=True)
                yield from bps.abs_set(self.spectrum_type, spectrum_type, wait=True)
                # yield from shutter_close()
                # yield from LED_on()
                yield from bps.mv(LED, 'High', UV_shutter, 'Low')
                yield from bps.sleep(2)
                uid = (yield from count([self], md=_md))
        
        yield from bps.mv(LED, 'Low', UV_shutter, 'Low')
        
        if csv_path!=None or plot==True:
            yield from bps.sleep(2)
            self.export_from_scan(uid, csv_path, sample_type, plot=plot, data_agent=data_agent)
        
    def export_from_scan(self, uid, csv_path, sample_type=None, plot=False, data_agent='db', wait=False):
        if wait==True:
            time.sleep(2)
        
        if data_agent == 'db':      
            unix_time = db[uid].start['time']     
            date, time = _readable_time(unix_time)
            
            x_axis_data = db[uid].table().QEPro_x_axis[1]
            output_data = db[uid].table().QEPro_output[1]
            sample_data = db[uid].table().QEPro_sample[1]
            dark_data = db[uid].table().QEPro_dark[1]
            reference_data = db[uid].table().QEPro_reference[1]
            spectrum_type = db[uid].table().QEPro_spectrum_type[1]
            int_time = db[uid].table().QEPro_integration_time[1]
            num_average = db[uid].table().QEPro_num_spectra[1]
            boxcar_width = db[uid].table().QEPro_buff_capacity[1]
            
            full_uid = db[uid].start['uid']

            metadata_list=['pumps','precursors','infuse_rate','infuse_rate_unit','pump_status',
                           'mixer','sample_type']
            if 'pumps' in db[uid].start.keys():
                pump_names = db[uid].start['pumps']
            else: pump_names = ['None']
            if 'precursors' in db[uid].start.keys():
                precursor = db[uid].start['precursors']
            else: precursor = ['None']
            if 'infuse_rate' in db[uid].start.keys():
                infuse_rate = db[uid].start['infuse_rate']
            else: infuse_rate = ['None']
            if 'infuse_rate_unit' in db[uid].start.keys():
                infuse_rate_unit = db[uid].start['infuse_rate_unit']
            else: infuse_rate_unit = ['None']
            if 'pump_status' in db[uid].start.keys():
                pump_status = db[uid].start['pump_status']
            else: pump_status = ['None']
            if 'mixer' in db[uid].start.keys():
                mixer = db[uid].start['mixer']
            else: mixer = ['None']
            if 'sample_type' in db[uid].start.keys():
                sample_type = db[uid].start['sample_type']
            # else: sample_type = None
            

        if data_agent == 'tiled':    
            run = tiled_client[uid]
            ds = run.primary.read()
            meta = run.metadata
            
            date, time = _readable_time(int(ds['time'][0]))
            x_axis_data = ds['QEPro_x_axis'].values[0]
            output_data = ds['QEPro_output'].values[0]
            sample_data = ds['QEPro_sample'].values[0]
            dark_data = ds['QEPro_dark'].values[0]
            reference_data = ds['QEPro_reference'].values[0]
            spectrum_type = ds['QEPro_spectrum_type'].values[0]
            int_time = ds['QEPro_integration_time'].values[0]
            num_average = ds['QEPro_num_spectra'].values[0]
            boxcar_width = ds['QEPro_buff_capacity'].values[0]
            
            full_uid = meta['start']['uid']

            if 'pumps' in meta['start'].keys():
                pump_names = meta['start']['pumps']
            else: pump_names = ['None']
            if 'precursors' in meta['start'].keys():
                precursor = meta['start']['precursors']
            else: precursor = ['None']
            if 'infuse_rate' in meta['start'].keys():
                infuse_rate = meta['start']['infuse_rate']
            else: infuse_rate = ['None']
            if 'infuse_rate_unit' in meta['start'].keys():
                infuse_rate_unit = meta['start']['infuse_rate_unit']
            else: infuse_rate_unit = ['None']
            if 'pump_status' in meta['start'].keys():
                pump_status = meta['start']['pump_status']
            else: pump_status = ['None']
            if 'mixer' in meta['start'].keys():
                mixer = meta['start']['mixer']
            else: mixer = ['None']
            if 'sample_type' in meta['start'].keys():
                sample_type = meta['start']['sample_type']
            # else: sample_type = None
             
        
        if plot == True:
            x_axis_label = self.x_axis_format.get(as_string=True)
            if spectrum_type == 3:
                y_axis_label = 'Absorbance'
            elif spectrum_type == 2:
                y_axis_label = 'Fluorescence'
            plt.figure()
            plt.plot(x_axis_data, output_data)
            plt.xlabel(x_axis_label)
            plt.ylabel(y_axis_label)
            plt.show()

        if csv_path != None:
            
            if spectrum_type == 3:
                spec = 'Abs'
                fout = f'{csv_path}/{sample_type}_{spec}_{date}-{time}_{uid[0:8]}.csv'
                
            if spectrum_type == 2:
                spec = 'PL'
                fout = f'{csv_path}/{sample_type}_{spec}_{date}-{time}_{uid[0:8]}.csv'

            with open(fout, 'w') as fp:
                fp.write(f'uid,{full_uid}\n')
                fp.write(f'Time_QEPro,{date},{time}\n')
                fp.write(f'Integration time (ms),{int_time}\n')
                fp.write(f'Number of averaged spectra,{num_average}\n')
                fp.write(f'Boxcar width,{boxcar_width}\n')

                for i in range(len(pump_names)):
                    fp.write(f'{pump_names[i]},{precursor[i]},{infuse_rate[i]},{infuse_rate_unit[i]},{pump_status[i]}\n')
            
                if mixer != None:
                    for i in range(len(mixer)):
                        fp.write(f'Mixer no. {i+1},{mixer[i]}\n')

                if spectrum_type == 3:
                    fp.write('Wavelength,Dark,Reference,Sample,Absorbance\n')
                else:
                    fp.write('Wavelength,Dark,Sample,Fluorescence\n')

                for i in range(len(output_data)):
                    if spectrum_type == 3:
                        fp.write(f'{x_axis_data[i]},{dark_data[i]},{reference_data[i]},{sample_data[i]},{output_data[i]}\n')
                    else:
                        fp.write(f'{x_axis_data[i]},{dark_data[i]},{sample_data[i]},{output_data[i]}\n')
        
# try:
#     from tiled.client import from_profile
#     tiled_client = from_profile("xpd-ldrd20-31")
# except (NameError, ModuleNotFoundError):
#     pass
# qepro = QEPro('XF:28ID2-ES{QEPro:Spec-1}:', name='QEPro', )
