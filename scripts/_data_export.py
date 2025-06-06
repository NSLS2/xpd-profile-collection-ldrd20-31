import datetime
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd
import os
import json
import importlib
# importlib.import_module("_data_analysis")
from _data_analysis import *

""" This module provides functions for data export.
    Usually imported as de.
"""

## Read input excel spreadsheet
def _read_input_xlsx(fn, sheet_name='inputs', skiprows=1, header=None, index_col=0):
    df = pd.read_excel(fn, sheet_name=sheet_name, skiprows=skiprows, header=header, index_col=index_col)
    input_dic = {}
    for i in df.index:
        if str(i) == 'nan':
            pass
        else:
            try:
                input_dic[i] = df.loc[i].dropna().tolist()
            except AttributeError:
                # temp = df.loc[i].to_dict(orient='split')
                input_dic[i] = []
                for j in range(df.loc[i].shape[0]):
                    ll = df.loc[i].iloc[j].dropna().tolist()
                    input_dic[i].append(ll)
    return input_dic




## dump or read glbl["_dark_dict_list"] into or from a json file
def _dark_json(_dark_dict_list, json_fn, dump_or_load='dump'):
    if dump_or_load == 'dump':
        with open(json_fn, 'w') as f:
            # indent=2 is not needed but makes the file human-readable 
            # if the data is nested
            json.dump(_dark_dict_list, f, indent=2)
            
        print(f'Dump dark_dict_list to {json_fn}')

    elif dump_or_load == 'load':
        with open(json_fn, 'r') as f:
            new_dark = json.load(f)
        for dark in new_dark:
            _dark_dict_list.append(dark)
        
        print(f'Append dark list from {os.path.basename(json_fn)} to _dark_dict_list')
        # return _dark_dict_list
        
    



## Creat file name based on date, time, uid, sample_name/type from fluorescence stream
def _fn_generator(uid, beamline_acronym='xpd', stream_name='fluorescence', data_agent='tiled'):
    _, metadata_dic = read_qepro_by_stream(uid, stream_name=stream_name, data_agent=data_agent, beamline_acronym=beamline_acronym)
    sample_type = metadata_dic['sample_type']
    date, ttime = _readable_time(metadata_dic['time'])
    full_uid = metadata_dic['uid']
    fn = f'{sample_type}_{date}-{ttime}_{full_uid[0:8]}'
    return fn


## Auto generate sample name with given prefix and infuse_rate
## If prefix = None, 'Pre00', 'Pre01', 'Pre02', ... will be used.
def _auto_name_sample(infuse_rates, prefix=None):
    infuse_rates = np.asarray(infuse_rates)

    if len(infuse_rates.shape) == 1:
        infuse_rates = infuse_rates.reshape(1, infuse_rates.shape[0])

    if prefix == None:
        prefix_list = [f'Pre{i:02d}' for i in range(infuse_rates.shape[1])]
    else:
        prefix_list = prefix
    
    # prefix_array = np.asarray(prefix_list)

    sample = []
    for i in range(infuse_rates.shape[0]):
        name = ''
        # for j in range(infuse_rates.shape[1]):
        for j in range(len(prefix_list)):
            int_rate = int(round(float(infuse_rates[i][j]), 0))
            name += f'{prefix_list[j]}_{int_rate:03d}_'
        sample.append(name[:-1])
    
    return sample



## Auto generate sample name with given prefix and directly reading pump rates
## If prefix = None, 'Pre00', 'Pre01', 'Pre02', ... will be used.
def _auto_name_sample2(pump_list, prefix=None):
    
    if type(pump_list[0]) == str:
        pump_list = [eval(pump) for pump in pump_list]
    else:
        pass

    get_rate = [pump.read_infuse_rate.get() for pump in pump_list]
    get_unit = [pump.read_infuse_rate_unit.get() for pump in pump_list]
    ruc_list = [rate_unit_converter(r0 = unit, r1 = 'ul/min') for unit in get_unit]
    get_status = [pump.status.get() for pump in pump_list]

    is_infusing = []
    for pump_status in get_status:
        if pump_status == 'Infusing':
            is_infusing.append(1)
        else:
            is_infusing.append(0)

    infuse_rates = np.asarray(get_rate) * np.asarray(ruc_list) * np.asarray(is_infusing)

    if len(infuse_rates.shape) == 1:
        infuse_rates = infuse_rates.reshape(1, infuse_rates.shape[0])

    if prefix == None:
        prefix_list = [f'Pre{i:02d}' for i in range(infuse_rates.shape[1])]
    else:
        prefix_list = prefix

    sample = []
    for i in range(infuse_rates.shape[0]):
        name = ''
        # for j in range(infuse_rates.shape[1]):
        for j in range(len(prefix_list)):
            int_rate = int(round(float(infuse_rates[i][j]), 0))
            name += f'{prefix_list[j]}_{int_rate:03d}_'
        sample.append(name[:-1])
    
    return sample



def _exprot_rate_agent(metadata_dic, rate_label_dic, agent_data):
    
    is_infusing = []
    for pump_status in metadata_dic['pump_status']:
        if pump_status == 'Infusing':
            is_infusing.append(1)
        elif pump_status == 'Idle':
            is_infusing.append(0)

    for key in rate_label_dic.keys():
        is_key_in_precursors = np.asarray([key in precursor for precursor in metadata_dic['precursors']])
        num_idx = np.argwhere(is_key_in_precursors==True)
        
        if num_idx.size == 1:
            pre_idx = num_idx[0][0]
            try:
                ruc = rate_unit_converter(r0 = metadata_dic["infuse_rate_unit"][pre_idx], r1 = 'ul/min')
                agent_data[rate_label_dic[key]] = metadata_dic["infuse_rate"][pre_idx]*ruc*is_infusing[pre_idx]
            except IndexError:
                agent_data[rate_label_dic[key]] = 0

        elif num_idx.size == 0:
            agent_data[rate_label_dic[key]] = 0

        elif num_idx.size > 1:
            raise ValueError (f'Check key: {key} of rate_label_dic. More than one precursor was found.')    

    return agent_data



def _readable_time(unix_time):
    from datetime import datetime
    dt = datetime.fromtimestamp(unix_time)
    # print(f'{dt.year}{dt.month:02d}{dt.day:02d},{dt.hour:02d}{dt.minute:02d}{dt.second:02d}')
    return (f'{dt.year}{dt.month:02d}{dt.day:02d}'), (f'{dt.hour:02d}{dt.minute:02d}{dt.second:02d}')


def _data_keys():
    qepro_list=['QEPro_x_axis', 'QEPro_output', 'QEPro_sample', 'QEPro_dark', 'QEPro_reference', 
                'QEPro_spectrum_type', 'QEPro_integration_time', 'QEPro_num_spectra', 'QEPro_buff_capacity']
    qepro_dic = {}

    metadata_list=['uid', 'time', 'pumps','precursors','infuse_rate','infuse_rate_unit',
                   'pump_status', 'mixer','sample_type', 'note']
    metadata_dic = {}

    return qepro_list, qepro_dic, metadata_list, metadata_dic


### the export funs below are revised from self.export_from_scan in 10-QEPro.py
def export_qepro_by_stream(uid, csv_path, stream_name='primary', data_agent='tiled', plot=False, wait=False):
    if wait==True:
        time.sleep(2)
    
    qepro_dic, metadata_dic = read_qepro_by_stream(uid, stream_name=stream_name, data_agent=data_agent)
    dic_to_csv_for_stream(csv_path, qepro_dic, metadata_dic, stream_name=stream_name)
    print(f'Export {stream_name} in uid: {uid[0:8]} to ../{os.path.basename(csv_path)} done!')
    

def read_qepro_by_stream(uid, stream_name='primary', data_agent='tiled', beamline_acronym='xpd-ldrd20-31'):
    
    if data_agent == 'db':
        try:
            catalog
        except NameError:
            import databroker
            catalog = databroker.catalog[beamline_acronym]
        run = catalog[uid]
        meta = run.metadata
    
    if data_agent == 'tiled':
        try:
            from_profile
        except NameError:
            from tiled.client import from_profile
            tiled_client = from_profile(beamline_acronym)
            
        run = tiled_client[uid]
        meta = run.metadata
   
    
    try:
        data = run[stream_name].read()
        qepro_list, qepro_dic, metadata_list, metadata_dic = _data_keys()

        for i in qepro_list:
            qepro_dic[i] = data[i].values

        for i in metadata_list:
            if i in meta['start'].keys():
                metadata_dic[i] = meta['start'][i]
            else:
                metadata_dic[i] = [None]
        metadata_dic['stream_name'] = stream_name

    except (KeyError, AttributeError):
        qepro_dic, metadata_dic = {}, {}
        print(f"Stream name: {stream_name} doesn't exist.")
    
    return qepro_dic, metadata_dic



## TODO: add reading scattering data
# def read_scatter_by_stream(uid, stream_name='scattering', data_agent='tiled', beamline_acronym='xpd-ldrd20-31'):
#     return scatter_dic, metadata_dic

## Get uids from data folders through daily experiments for export database for ML
## Read uid from **fluorescence** folder
def get_daily_uids(data_folder_list):
    uid_list = []
    for data_folder in data_folder_list:
        subfolder = glob.glob(data_folder + '/**fluorescence')
        subfolder.sort()

        for folder in subfolder:
            fn = glob.glob(folder + '/**csv')[-1]
            with open(fn, 'r') as f:
                uid_ = f.readlines()[0]
                uid = uid_.strip('\n').split(',')[-1]
                uid_list.append(uid)
            f.close()
    return uid_list



def dic_to_csv_for_stream(csv_path, qepro_dic, metadata_dic, stream_name='primary', fitting=None, plqy_dic=None):
    # to save fitting results for good data, fitting needs be a dict with two keys:
    # fitting = {'fit_function': da._1gauss, 'curve_fit': popt}

    spectrum_type = qepro_dic['QEPro_spectrum_type']
    sample_type = metadata_dic['sample_type']
    date, time = _readable_time(metadata_dic['time'])
    full_uid = metadata_dic['uid']
    int_time = qepro_dic['QEPro_integration_time']
    num_average = qepro_dic['QEPro_num_spectra']
    boxcar_width = qepro_dic['QEPro_buff_capacity']
    pump_names = metadata_dic['pumps']
    precursor = metadata_dic['precursors']
    infuse_rate = metadata_dic['infuse_rate']
    infuse_rate_unit = metadata_dic['infuse_rate_unit']
    pump_status = metadata_dic['pump_status']
    mixer = metadata_dic['mixer']
    note = metadata_dic['note']

    x_axis_data = qepro_dic['QEPro_x_axis']
    dark_data = qepro_dic['QEPro_dark']
    reference_data = qepro_dic['QEPro_reference']
    sample_data = qepro_dic['QEPro_sample']
    output_data = qepro_dic['QEPro_output']

    try:
        f1 = fitting['fit_function']
        popt = fitting['curve_fit']
        fitted_y = f1(x_axis_data[-1], *popt)
        # output_mean = np.mean(output_data, axis=0)
        output_mean = fitting['percentile_mean']  ## add percentile filtering mean data here
    except (TypeError, KeyError):
        if fitting == None:
            pass
        else:
            print('Input fitting info is not correct.\n'
                  'Please fllow as:  {"fit_function": da._1gauss, "curve_fit": popt}')

    if (stream_name == 'take_a_uvvis' or stream_name == 'primary') and fitting == None:
        if spectrum_type == 3:
            spec = 'Abs'
            fout = f'{csv_path}/{sample_type}_{spec}_{date}-{time}_{full_uid[0:8]}.csv'
            
        elif spectrum_type == 2:
            spec = 'PL'
            fout = f'{csv_path}/{sample_type}_{spec}_{date}-{time}_{full_uid[0:8]}.csv'

        with open(fout, 'w') as fp:
            fp.write(f'uid,{full_uid}\n')
            fp.write(f'Time_QEPro,{date},{time}\n')
            fp.write(f'Integration time (ms),{int_time[0]}\n')
            fp.write(f'Number of averaged spectra,{num_average[0]}\n')
            fp.write(f'Boxcar width,{boxcar_width[0]}\n')

            for i in range(len(pump_names)):
                fp.write(f'{pump_names[i]},{precursor[i]},{infuse_rate[i]},{infuse_rate_unit[i]},{pump_status[i]}\n')
        
            if mixer != None:
                for i in range(len(mixer)):
                    fp.write(f'Mixer no. {i+1},{mixer[i]}\n')

            if type(note) is str:
                fp.write(f'Note,{note}\n')

            if spectrum_type == 3:
                fp.write('Wavelength,Dark,Reference,Sample,Absorbance\n')
            else:
                fp.write('Wavelength,Dark,Sample,Fluorescence\n')

            for i in range(x_axis_data.shape[1]):
                if spectrum_type == 3:
                    fp.write(f'{x_axis_data[0,i]},{dark_data[0,i]},{reference_data[0,i]},{sample_data[0,i]},{output_data[0,i]}\n')
                else:
                    fp.write(f'{x_axis_data[0,i]},{dark_data[0,i]},{sample_data[0,i]},{output_data[0,i]}\n')
    
    elif (stream_name != 'take_a_uvvis' and stream_name != 'primary') and fitting == None:
        new_dir = f'{csv_path}/{date}{time}_{full_uid[0:8]}_{stream_name}'
        os.makedirs(new_dir, exist_ok=True)
        for j in range(x_axis_data.shape[0]):
            fout = f'{new_dir}/{sample_type}_{date}-{time}_{full_uid[0:8]}_{j:03d}.csv'
            
            with open(fout, 'w') as fp:
                fp.write(f'uid,{full_uid}\n')
                fp.write(f'Time_QEPro,{date},{time}\n')
                fp.write(f'Integration time (ms),{int_time[j]}\n')
                fp.write(f'Number of averaged spectra,{num_average[j]}\n')
                fp.write(f'Boxcar width,{boxcar_width[j]}\n')

                for i in range(len(pump_names)):
                    fp.write(f'{pump_names[i]},{precursor[i]},{infuse_rate[i]},{infuse_rate_unit[i]},{pump_status[i]}\n')
            
                if mixer != None:
                    for i in range(len(mixer)):
                        fp.write(f'Mixer no. {i+1},{mixer[i]}\n')

                if type(note) is str:
                    fp.write(f'Note,{note}\n')

                if spectrum_type[0] == 3:
                    fp.write('Wavelength,Dark,Reference,Sample,Absorbance\n')
                else:
                    fp.write('Wavelength,Dark,Sample,Fluorescence\n')

                for i in range(x_axis_data.shape[1]):
                    if spectrum_type[0] == 3:
                        fp.write(f'{x_axis_data[j,i]},{dark_data[j,i]},{reference_data[j,i]},{sample_data[j,i]},{output_data[j,i]}\n')
                    else:
                        fp.write(f'{x_axis_data[j,i]},{dark_data[j,i]},{sample_data[j,i]},{output_data[j,i]}\n')
    
    elif type(fitting) is dict:
        if (stream_name == 'take_a_uvvis') or (stream_name == 'primary'):
            fout = f'{csv_path}/{sample_type}_PL_{date}-{time}_{full_uid[0:8]}_fitted.csv'
        
        elif stream_name == 'fluorescence' or stream_name == 'absorbance':
            new_dir = f'{csv_path}/{date}{time}_{full_uid[0:8]}_{stream_name}'
            os.makedirs(new_dir, exist_ok=True)
            fout = f'{new_dir}/{sample_type}_{date}-{time}_{full_uid[0:8]}_fitted.csv'

        with open(fout, 'w') as fp:
            fp.write(f'uid,{full_uid}\n')
            fp.write(f'Time_QEPro,{date},{time}\n')
            fp.write(f'Integration time (ms),{int_time[0]}\n')
            fp.write(f'Number of averaged spectra,{num_average[0]}\n')
            fp.write(f'Boxcar width,{boxcar_width[0]}\n')

            for i in range(len(pump_names)):
                fp.write(f'{pump_names[i]},{precursor[i]},{infuse_rate[i]},{infuse_rate_unit[i]},{pump_status[i]}\n')
        
            if mixer != None:
                for i in range(len(mixer)):
                    fp.write(f'Mixer no. {i+1},{mixer[i]}\n')

            if type(note) is str:
                fp.write(f'Note,{note}\n')
            
            try:
                fun_name = f1.__name__
                fp.write(f'fitting function,{fun_name}\n')
            except (AttributeError, TypeError):
                pass

            fp.write('popt')
            for i in range(len(popt)):
                fp.write(f',{popt[i]}')
            fp.write('\n')

            try: 
                plqy = plqy_dic['plqy']
                PL_integral = plqy_dic['PL_integral']
                Absorbance_365 = plqy_dic['Absorbance_365']
                fp.write(f'plqy,{plqy},PL_integral,{PL_integral},Absorbance_365,{Absorbance_365}\n')
            except (TypeError, KeyError):
                pass

            # fp.write('Wavelength,Dark,Sample,Fluorescence_mean,Fitting\n')
            # for i in range(x_axis_data.shape[1]):
            #     fp.write(f'{x_axis_data[-1,i]},{dark_data[-1,i]},{sample_data[-1,i]},{output_mean[i]},{fitted_y[i]}\n')

            if spectrum_type[0] == 3:
                fp.write('Wavelength,Dark,Reference,Sample,Absorbance_mean,Offset\n')
            else:
                fp.write('Wavelength,Dark,Sample,Fluorescence_mean,Fitting\n')

            for i in range(x_axis_data.shape[1]):
                if spectrum_type[0] == 3:
                    fp.write(f'{x_axis_data[-1,i]},{dark_data[-1,i]},{reference_data[-1,i]},{sample_data[-1,i]},{output_mean[i]},{output_mean[i]-fitted_y[i]}\n')
                else:
                    fp.write(f'{x_axis_data[-1,i]},{dark_data[-1,i]},{sample_data[-1,i]},{output_mean[i]},{fitted_y[i]}\n')

        



def export_qepro_from_uid(uid, csv_path, sample_type=None, plot=False, data_agent='db', wait=False):
    if wait==True:
        time.sleep(2)
    
    if data_agent == 'db':
        qepro_dic, metadata_dic = read_qepro_from_db(uid)
    elif data_agent == 'tiled':
        qepro_dic, metadata_dic = read_qepro_from_tiled(uid)

    dic_to_csv(csv_path, qepro_dic, metadata_dic)
    print(f'Export uid: {uid[0:8]} to ../{os.path.basename(csv_path)} done!')



def read_qepro_from_tiled(uid):
    try:
       from_profile
    except NameError:
        from tiled.client import from_profile
        # tiled_client = from_profile("xpd-ldrd20-31")
        tiled_client = from_profile("xpd")  
    
    run = tiled_client[uid]
    ds = run.primary.read()
    meta = run.metadata
    
    # date, time = _readable_time(ds['time'][0])
    # full_uid = meta['start']['uid']
    
    qepro_list, qepro_dic, metadata_list, metadata_dic = _data_keys()

    for i in qepro_list:
        qepro_dic[i] = ds[i].values[0]

    for i in metadata_list:
        if i in meta['start'].keys():
            metadata_dic[i] = meta['start'][i]
        else:
            metadata_dic[i] = [None]
    
    return qepro_dic, metadata_dic



def read_qepro_from_db(uid):
    try:
        db
    except NameError:
        import databroker
        # db = databroker.Broker.named('xpd-ldrd20-31')
        db = databroker.Broker.named('xpd')
    
    # full_uid = db[uid].start['uid']
    # unix_time = db[uid].start['time']     
    # date, time = _readable_time(unix_time)

    qepro_list, qepro_dic, metadata_list, metadata_dic = _data_keys()

    for i in qepro_list:
        qepro_dic[i] = db[uid].table()[i][1]

    for i in metadata_list:
        if i in db[uid].start.keys():
            metadata_dic[i] = db[uid].start[i]
        else:
            metadata_dic[i] = [None]
    
    return qepro_dic, metadata_dic




def dic_to_csv(csv_path, qepro_dic, metadata_dic):
        
    spectrum_type = qepro_dic['QEPro_spectrum_type']
    sample_type = metadata_dic['sample_type']
    date, time = _readable_time(metadata_dic['time'])
    full_uid = metadata_dic['uid']
    int_time = qepro_dic['QEPro_integration_time']
    num_average = qepro_dic['QEPro_num_spectra']
    boxcar_width = qepro_dic['QEPro_buff_capacity']
    pump_names = metadata_dic['pumps']
    precursor = metadata_dic['precursors']
    infuse_rate = metadata_dic['infuse_rate']
    infuse_rate_unit = metadata_dic['infuse_rate_unit']
    pump_status = metadata_dic['pump_status']
    mixer = metadata_dic['mixer']
    note = metadata_dic['note']

    x_axis_data = qepro_dic['QEPro_x_axis']
    dark_data = qepro_dic['QEPro_dark']
    reference_data = qepro_dic['QEPro_reference']
    sample_data = qepro_dic['QEPro_sample']
    output_data = qepro_dic['QEPro_output']
    
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

        if type(note) is str:
            fp.write(f'Note,{note}\n')

        if spectrum_type == 3:
            fp.write('Wavelength,Dark,Reference,Sample,Absorbance\n')
        else:
            fp.write('Wavelength,Dark,Sample,Fluorescence\n')

        for i in range(len(output_data)):
            if spectrum_type == 3:
                fp.write(f'{x_axis_data[i]},{dark_data[i]},{reference_data[i]},{sample_data[i]},{output_data[i]}\n')
            else:
                fp.write(f'{x_axis_data[i]},{dark_data[i]},{sample_data[i]},{output_data[i]}\n')




## Convert ML parameters (non-dimensional) into device parameters (infuse rate)
## Note: Before covertion, it is recommended take a dummy scan 
##       to let BlueSky know the precursor list and total flow rate
def ML_to_device_parameters(ML_dof, metadata_dic, unit = 'ul/min', 
                            check_sum = True, check_negative = True, 
                            exp_name='ldrd-2031'):
    
    ML_dof = np.float32(ML_dof)

    _is_sum_equal_one = (ML_dof.sum() == 1.0)
    _is_any_negative = (len(np.argwhere(ML_dof<0)) > 0)

    if check_sum:
        if _is_sum_equal_one == False:
            raise ValueError(f'Sum of {ML_dof} is not equal to one.')


    if check_negative:
        if _is_any_negative == True:
            raise ValueError(f'Found negative value in {ML_dof}.')
    
    
    ## Get data from metadata_dic
    pump_status = metadata_dic['pump_status']
    precursors = metadata_dic['precursors']
    infuse_rate = np.float32(metadata_dic['infuse_rate'])
    infuse_rate_unit = metadata_dic['infuse_rate_unit']

    ## Turn pump_status into boolean array
    pump_status = list(map(lambda status: status.lower().capitalize() == "Infusing", pump_status))
    pump_status = np.float32(pump_status)

    ## Generate ruc (rate unit converter) to unify rate units as assigned unit
    ruc = []
    for i in range(len(infuse_rate_unit)):
        rr = rate_unit_converter(r0=infuse_rate_unit[i], r1=unit)
        ruc.append(rr)
    ruc = np.float32(ruc)
    
    unified_rate = infuse_rate * pump_status * ruc
    predicted_rate = np.round(ML_dof*unified_rate.sum(), decimals=2)

    ## Arrange ML_dof into an array as infuse_rate / infuse_rate_unit
    if exp_name == 'ldrd-2031':
        input_names = ['CsPb', 'Br', 'ZnI', 'ZnCl', 'Toluene']
    else:
        input_names = exp_name

    device_parameters = np.zeros(len(infuse_rate))
    for i in range(len(input_names)):
        for j in range(len(precursors)):
            if input_names[i].lower() in precursors[j].lower():
                device_parameters[j] = predicted_rate[i]
    
    return device_parameters





## Convert device parameters (infuse rate) into ML parameters (non-dimensional)
def device_to_ML_parameters(metadata_dic, exp_name='ldrd-2031'):
    
    ## Get data from metadata_dic
    pump_status = metadata_dic['pump_status']
    precursors = metadata_dic['precursors']
    infuse_rate = np.float32(metadata_dic['infuse_rate'])
    infuse_rate_unit = metadata_dic['infuse_rate_unit']

    ## Turn pump_status into boolean array
    pump_status = list(map(lambda status: status.lower().capitalize() == "Infusing", pump_status))    
    pump_status = np.float32(pump_status)

    ## Generate ruc (rate unit converter) to unify rate units as infuse_rate_unit[0]
    ruc = []
    for i in range(len(infuse_rate_unit)):
        rr = rate_unit_converter(r0=infuse_rate_unit[i], r1=infuse_rate_unit[0])
        ruc.append(rr)
    ruc = np.float32(ruc)
    
    unified_rate = infuse_rate * pump_status * ruc
    normalized_rate = unified_rate / unified_rate.sum()
    
    
    ## Arrange normalized_rate into an array as input_names
    if exp_name == 'ldrd-2031':
        input_names = ['CsPb', 'Br', 'ZnI', 'ZnCl', 'Toluene']
    else:
        input_names = exp_name
    
    ML_parameters = np.zeros(len(input_names))
    for i in range(len(input_names)):
        for j in range(len(precursors)):
            if input_names[i].lower() in precursors[j].lower():
                ML_parameters[i] = normalized_rate[j]
            
    return ML_parameters
