import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import importlib
import time
import pprint
import json
from scipy import integrate

sq = importlib.import_module("_synthesis_queue_RM")
de = importlib.import_module("_data_export")
da = importlib.import_module("_data_analysis")
pc = importlib.import_module("_pdf_calculator")

from diffpy.pdfgetx import PDFConfig
gp = importlib.import_module("_get_pdf")

build_agent = importlib.import_module("prepare_agent_pdf").build_agent
import torch

from tiled.client import from_uri
from bluesky_queueserver_api.zmq import REManagerAPI
from bluesky_queueserver_api import BPlan, BInst


""" This module provides class for Qsever and Kafka.
    Usually imported as LK.
"""

def _qserver_inputs():
    """Define namespace for Qserver inputs

    Returns:
        list: list for Qserver inputs
    All the namesapce in this list should be found in the excel spreadsheet.
    """
    qserver_list=[
            'zmq_control_addr', 'zmq_info_addr', 
            'dummy_qserver', 'is_iteration', 'pos', 'use_OAm', 
            'name_by_prefix', 'prefix', 'pump_list', 'precursor_list', 
            'syringe_mater_list', 'syringe_list', 'target_vol_list', 
            'sample', 
            'wait_dilute', 'if_wash', 'wash_loop', 'wash_sapphire', 
            'mixer', 'wash_tube', 'resident_t_ratio', 
            'rate_unit', 'uvvis_config', 'perkin_config', 
            'auto_set_target_list', 'set_target_list', 'infuse_rates', 
            ]

    return qserver_list


def _kafka_inputs():
    """Define namespace for Kafka inputs

    Returns:
        list: list for Kafka inputs
    All the namesapce in this list should be found in the excel spreadsheet.
    """
    inputs_list=[
            'dummy_kafka', 'csv_path', 'key_height', 'height', 'distance', 'PLQY', 
            'rate_label_dic_key', 'rate_label_dic_value', 'new_points_label', 
            'use_good_bad', 'post_dilute', 'fix_Br_ratio', 
            'write_agent_data', 'agent_data_path', 'build_agent', 
            'use_1st_prediction', 'USE_AGENT_iterate', 'peak_target',  
            'iq_to_gr', 'iq_to_gr_path', 'cfg_fn', 'bkg_fn', 'iq_fn',  
            'search_and_match', 'mystery_path', 'results_path', 
            'fitting_pdf', 'fitting_pdf_path', 'cif_fn', 'gr_fn', 
            'dummy_pdf', 'write_to_sandbox', 'sandbox_uri', 'beamline_acronym', 
            'fn_TBD', 
            ]

    return inputs_list



def _kafka_process():
    """ Define namespace for variables processed in Kafka

    Returns:
        list: list for variables processed in Kafka
    All the namesapce in this list should not be found in the excel spreadsheet.
    """
    process_list=[ 
            'agent', 'sandbox_tiled_client', 'tiled_client', 
            'entry', 'iq_data', 'stream_list', 
            'uid', 'uid_catalog', 'uid_bundle', 'uid_pdfstream', 'uid_sandbox', 
            'gr_data', 'pdf_property', 'gr_fitting', 
            'qepro_dic', 'metadata_dic', 'sample_type', 
            'PL_goodbad', 'PL_fitting', 'abs_data', 'abs_fitting', 
            'plqy_dic', 'optical_property', 'agent_data', 'rate_label_dic', 
            'good_data', 'bad_data', 'continue_iteration', 'finished', 
            
            ]

    return process_list




class dic_to_inputs():
    def __init__(self, parameters_dict, parameters_list):
        """ Turn the input dict into class attributes

        Args:
            parameters_dict (dict): dict read from excel spreadsheet
            parameters_list (list): list for namespace deined above
        """
        self.parameters_dict = parameters_dict
        self.parameters_list = parameters_list

        for key in self.parameters_list:
            # print(f'{key = }')
            try:
                setattr(self, key, self.parameters_dict[key])
            except KeyError:
                setattr(self, key, [])
                print(f'{key = } not in parameters_dict so set to empty list.')


class xlsx_to_inputs():
    def __init__(self, parameters_list, xlsx_fn, sheet_name='inputs', is_kafka=False):
        """ Read the excel spreadsheet according to the sheet name into a dict
        Turn the dict read from excel into class attributes

        Args:
            parameters_list (list): list for namespace deined above
            xlsx_fn (str): full path if excel file
            sheet_name (str, optional): sheet name of the excel file. Defaults to 'inputs'.
            is_kafka (bool, optional): if True, the namespace for the variables processed 
                                        in Kafka will be turn into class attributes. Defaults to False.
        """
        self.parameters_list = parameters_list
        self.from_xlsx = xlsx_fn
        self.sheet_name = sheet_name

        ## set attributes of keys in self.parameters_list for input parameters
        self.print_dic = de._read_input_xlsx(self.from_xlsx, sheet_name=self.sheet_name)
        ## Every attribute in self.inputs is default to a list!!!
        self.inputs = dic_to_inputs(self.print_dic, self.parameters_list)

        
        if is_kafka:
            ## set attributes of keys in _kafka_process() for processing data
            for key in _kafka_process():
                setattr(self, key, [])
            
            try:
                if self.inputs.build_agent[0]: 
                    ## Assign Agent to self.agent
                    self.agent
                    print('\n***** Start to initialize blop agent ***** \n')
                    self.agent = build_agent(
                            peak_target=self.inputs.peak_target[0], 
                            agent_data_path=self.inputs.agent_data_path[0])
                    print(f'\n***** Initialized blop agent at {self.agent} ***** \n')

                ## self.inputs.sandbox_uri[0] is just the uri of sandbox
                ## so, turn uri into client and assign it to self.sandbox_tiled_client
                if type(self.inputs.sandbox_uri[0]) is str:
                    self.sandbox_tiled_client = from_uri(self.inputs.sandbox_uri[0])


                ## Use glob.glob to find the complete path of cfg_fn, bkg_fn, iq_fn, cif_fn, gr_fn
                # fn_TBD = ['cfg_fn', 'bkg_fn', 'iq_fn', 'cif_fn', 'gr_fn']
                for fn in self.inputs.fn_TBD:
                    
                    path = getattr(self.inputs, fn)[0]
                    if path in self.parameters_list:
                        path = getattr(self.inputs, path)[0]
                    
                    ff = getattr(self.inputs, fn)[1]

                    fn_glob = glob.glob(os.path.join(path, ff))

                    for i in fn_glob:
                        getattr(self.inputs, fn).append(i)

                ## Making rate_label_dic by rate_label_dic_key and rate_label_dic_value
                self.rate_label_dic = {}
                for key, value in zip(self.inputs.rate_label_dic_key, self.inputs.rate_label_dic_value):
                    self.rate_label_dic.update({key: value})
            
            except AttributeError:
                pass


    def save_kafka_dict(self, home_path, reset_uid_catalog=True):
        """ Save the selected variables processed in Kafka into a jason

        Args:
            home_path (str): directory to save json for Kafka dict
            reset_uid_catalog (bool, optional): if True, the selected varaibles will be reset to empty lists. Defaults to True.
        """

        date, ttime = de._readable_time(self.metadata_dic['time'])
        json_fn = f'{date}-{ttime}_{self.uid[0:8]}.json'

        ## Make directory for home_path folder
        try:
            os.mkdir(home_path)
        except FileExistsError:
            pass

        json_path = os.path.join(home_path, json_fn)

        key_to_save = [ 
                'uid', 'uid_catalog', 'uid_bundle', 'uid_pdfstream', 'uid_sandbox', 
                'metadata_dic', 'pdf_property', 'optical_property', 
                'agent_data', 'continue_iteration', 'finished', ]
        
        kafka_process_dict = {}
        for key in key_to_save:
            kafka_process_dict.update({key: getattr(self, key)})

        with open(json_path, "w") as f:
            json.dump(kafka_process_dict, f, indent=2)
            json.dump(self.print_dic, f, indent=2)

        print(f"\nwrote kafka info to {home_path}\n")

        if reset_uid_catalog:
            self.reset_kafka_process(['uid_catalog'])
    
    
    
    ## Reset attributes of key in keys to empty lists for next event
    def reset_kafka_process(self, keys):
        for key in keys:
            setattr(self, key, [])

    
    
    def auto_rate_list(self, pump_list, new_points, fix_Br_ratio):
        """Auto transfer the predicted rates in new_points to a rate_list for qserver

        Args:
            pump_list (list): pump list for qserver
            new_points (dict): new_points predicted by self.agent
            fix_Br_ratio (list): if fix ratio of CsPb/Br. If fixed, the ratio is fix_Br_ratio[1]

        Returns:
            list: rate list for importing to qserver
        """
        set_target_list = [0 for i in range(len(pump_list))]
        rate_list = []
        for i in self.inputs.new_points_label:
            if i in new_points['points'].keys():
                rate_list.append(new_points['points'][i][0])
            else:
                pass
        
        if fix_Br_ratio[0]:
            rate_list.insert(1, rate_list[0]*fix_Br_ratio[1])
        
        return rate_list



    def macro_agent(self, qserver_process, RM, check_target=False, is_1st=False):
        """macro to build agent, make optimization, and update agent_data

        This macro will
        1. Build agent from agent_data_path = self.inputs.agent_data_path[0]
        2. Make optimization
        2. Update self.agent_data with target, predicted mean & standard deviation
            self.agent_data['agent_target']:       agent_target
            self.agent_data['posterior_mean']:     post_mean
            self.agent_data['posterior_stddev']:   post_stddev
        3. Check if meet target. If meet, wash loop; if not, keep iteration.
        4. Update self.continue_iteration

        Args:
            qserver_process (_LDRD_Kafka.xlsx_to_inputs, optional): qserver parameters read from xlsx.
            RM (REManagerAPI): Run Engine Manager API.
            check_target (bool, optional): Check if peak emission reaches peak target. Defaults to False.
            is_ist (bool, optional): Check if it is the first precidciton. If yes, skip build agent.

        Returns:
            dict: new_points predicted by self.agent
        """

        qin = qserver_process.inputs
        peak_target = self.inputs.peak_target[0]
        peak_tolerance = self.inputs.peak_target[1]


        if check_target:
            peak_diff = abs(self.PL_fitting['peak_emission'] - peak_target)
            meet_target = (peak_diff <= peak_tolerance)
            if meet_target:
                print(f'\nTarget peak: {self.inputs.peak_target[0]} nm vs. Current peak: {self.PL_fitting["peak_emission"]} nm\n')
                print(f'\nReach the target, stop iteration, stop all pumps, and wash the loop.\n')

                ### Stop all infusing pumps and wash loop
                sq.wash_tube_queue2(qin.pump_list, qin.if_wash, qin.wash_loop, 'ul/min', 
                                zmq_control_addr=qin.zmq_control_addr[0],
                                zmq_info_addr=qin.zmq_info_addr[0])
                
                inst1 = BInst("queue_stop")
                RM.item_add(inst1, pos='front')
                self.continue_iteration.append(False)
        
        else:
            self.continue_iteration.append(True)       

        # if self.inputs.build_agent[0]: 
        if is_1st:
            pass
        else:
            self.agent = build_agent(
                            peak_target = peak_target, 
                            agent_data_path = self.inputs.agent_data_path[0], 
                            use_OAm = qin.use_OAm[0])

        if len(self.agent.table) < 2:
            acq_func = "qr"
        else:
            acq_func = "qem"
        
        new_points = self.agent.ask(acq_func, n=1)

        ## Get target of agent.ask()
        agent_target = self.agent.objectives.summary['target'].tolist()
        
        ## Get mean and standard deviation of agent.ask()
        res_values = []
        for i in self.inputs.new_points_label:
            if i in new_points['points'].keys():
                res_values.append(new_points['points'][i][0])
        x_tensor = torch.tensor(res_values)
        posterior = self.agent.posterior(x_tensor)
        post_mean = posterior.mean.tolist()[0]
        post_stddev = posterior.stddev.tolist()[0]

        ## apply np.exp for log-transform objectives
        if_log = self.agent.objectives.summary['transform']
        for j in range(if_log.shape[0]):
            if if_log[j] == 'log':
                post_mean[j] = np.exp(post_mean[j])
                post_stddev[j] = np.exp(post_stddev[j])

        ## Update target, mean, and standard deviation in agent_data
        self.agent_data = {}
        self.agent_data.update({'agent_target': agent_target})
        self.agent_data.update({'posterior_mean': post_mean})
        self.agent_data.update({'posterior_stddev': post_stddev})
        
        return new_points
    
    
    
    def macro_00_print_start(self, message):
        """macro to print metadata when doc name is start and reset self.uid to an empty list

        Args:
            message (dict): message in RE document
        """
        
        if 'uid' in message.keys():
            print(f"uid: {message['uid']}")
        if 'plan_name' in message.keys():
            print(f"plan name: {message['plan_name']}")
        if 'detectors' in message.keys(): 
            print(f"detectors: {message['detectors']}")
        if 'pumps' in message.keys(): 
            print(f"pumps: {message['pumps']}")
        if 'detectors' in message.keys(): 
            print(f"detectors: {message['detectors']}")
        if 'uvvis' in message.keys() and (len(message['uvvis'])==3):
            print(f"uvvis mode:\n"
                    f"           integration time: {message['uvvis'][0]} ms\n"
                    f"           num spectra averaged: {message['uvvis'][1]}\n"
                    f"           buffer capacity: {message['uvvis'][2]}"
                    )
        elif 'uvvis' in message.keys() and (len(message['uvvis'])==5):
            print(f"uvvis mode:\n"
                    f"           spectrum type: {message['uvvis'][0]}\n"
                    f"           integration time: {message['uvvis'][2]} ms\n"
                    f"           num spectra averaged: {message['uvvis'][3]}\n"
                    f"           buffer capacity: {message['uvvis'][4]}"
                    )                
        if 'mixer' in message.keys():
            print(f"mixer: {message['mixer']}")
        if 'sample_type' in message.keys():
            print(f"sample type: {message['sample_type']}")
        
        ## Reset self.uid to an empty list
        self.uid = []
    



    def macro_01_stop_queue_uid(self, RM, message):
        """macro to stop queue and get raw data uid, used in kafka consumer
        while taking a Uv-Vis, no X-ray data but still do analysis of pdfstream
        
        This macro will
        1. Stop queue
        2. Assign raw data uid to self.uid
        3. Append raw data uid to self.uid_catalog
        4. Update self.stream_list

        Args:
            RM (REManagerAPI): Run Engine Manager API.
            message (dict): message in RE document
        """
        inst1 = BInst("queue_stop")
        RM.item_add(inst1, pos='front')
        ## wait 1 second for databroker to save data
        time.sleep(1)
        self.uid = message['run_start']
        self.uid_catalog.append(self.uid)
        stream_list = list(message['num_events'].keys())
        ## Reset self.stream_list to an empty list
        self.stream_list = []
        for stream_name in stream_list:
            self.stream_list.append(stream_name)




    def macro_02_get_iq(self, iq_I_uid):
        """macro to get iq data, used in kafka consumer 
        whiel taking xray_uvvis_plan and analysis of pdfstream finished
       
        This macro will
        0. Reset self.iq_data as an empty {}
        1. Assgin sandbox entry to self.entry
        2. Update iq_data as a dict into self.iq_data
            self.iq_data['Q']:     chi_Q
            self.iq_data['I']:     chi_I
            self.iq_data['array']: np.array([chi_Q, chi_I]) 
            self.iq_data['df']:    pd.DataFrame([chi_Q, chi_I])
        3. Reset self.uid to an empty list

        Args:
            iq_I_uid (str): uid of analysis data, read from doc[1]['data']['chi_I']

        """
        self.uid_pdfstream.append(iq_I_uid)
        self.entry = self.sandbox_tiled_client[iq_I_uid]
        df = self.entry.read()
        # Before updating I(Q) data, reset self.iq_data as an empty dict
        self.iq_data = {}
        # self.iq_data.append(df['chi_Q'].to_numpy())
        # self.iq_data.append(df['chi_I'].to_numpy())

        iq_array = np.asarray([df['chi_Q'].to_numpy(), df['chi_I'].to_numpy()])
        # self.iq_data.append(iq_array)

        iq_df = pd.DataFrame()
        iq_df['q'] = df['chi_Q'].to_numpy()
        iq_df['I(q)'] = df['chi_I'].to_numpy()
        # self.iq_data.append(iq_df)
        
        iq_data = { 'Q':df['chi_Q'].to_numpy(), 
                    'I':df['chi_I'].to_numpy(), 
                    'array':iq_array, 
                    'df':iq_df}
        self.iq_data.update(iq_data)

        ## Reset self.uid to an empty list
        self.uid = []



    def macro_03_get_uid(self):
        """macro to get raw data uid, used in kafka consumer
       
        This macro will
        1. Assign raw data uid to self.uid
        2. Append raw data uid to self.uid_catalog
        3. Update self.stream_list
        """
        ## wait 1 second for databroker to save data
        time.sleep(1)
        self.uid = self.entry.metadata['run_start']
        self.uid_catalog.append(self.uid)
        stream_list = self.tiled_client[self.uid].metadata['summary']['stream_names']
        ## Reset self.stream_list to an empty list
        self.stream_list = []
        for stream_name in stream_list:
            self.stream_list.append(stream_name)




    def macro_04_dummy_pdf(self):
        """macro to setup a dummy pdf data for testing, used in kafka consumer
        while self.inputs.dummy_pdf[0] is True

        This macro will
        0. Reset self.iq_data as an empty dict
        1. Read pdf data from self.iq_fn[-1]
        2. Update iq_data as a dict into self.iq_data
            self.iq_data['Q']:     iq_array[0]
            self.iq_data['I']:     iq_array[1]
            self.iq_data['array']: iq_array
            self.iq_data['df']:    iq_df
        """
        self.iq_data = {}
        iq_array = pd.read_csv(self.inputs.iq_fn[-1], skiprows=1, names=['q', 'I(q)'], sep=' ').to_numpy().T
        # self.iq_data.append(iq_array[0])
        # self.iq_data.append(iq_array[1])
        # self.iq_data.append(iq_array)
        iq_df = pd.read_csv(self.inputs.iq_fn[-1], skiprows=1, names=['q', 'I(q)'], sep=' ')
        # self.iq_data.append(iq_df)

        iq_data = { 'Q':iq_array[0], 
                    'I':iq_array[1], 
                    'array':iq_array, 
                    'df':iq_df}
        self.iq_data.update(iq_data)



    def macro_05_iq_to_gr(self, beamline_acronym):
        """macro to condcut data reduction from I(Q) to g(r), used in kafka consumer
        
        This macro will
        1. Generate a filename for g(r) data by using metadata of stream_name == fluorescence
        2. Read pdf config file from self.inputs.cfg_fn[-1]
        3. Read pdf background file from self.inputs.bkg_fn[-1]
        4. Generate s(q), f(q), g(r) data by gp.transform_bkg() and save in self.inputs.iq_to_gr_path[0]
        5. Read saved g(r) into pd.DataFrame and save again to remove the headers
        6. Update g(r) data path and data frame to self.gr_data
            self.gr_data[0]: gr_data (path)
            self.gr_data[1]: gr_df

        Args:
            beamline_acronym (str): catalog name for tiled to access data
        """
        # Grab metadat from stream_name = fluorescence for naming gr file
        fn_uid = de._fn_generator(self.uid, beamline_acronym=beamline_acronym)
        gr_fn = f'{fn_uid}_scattering.gr'

        ### dummy test, e.g., CsPbBr2
        if self.inputs.dummy_pdf[0]:
            gr_fn = f'{self.inputs.iq_fn[-1][:-4]}.gr'

        # Build pdf config file from a scratch
        pdfconfig = PDFConfig()
        pdfconfig.readConfig(self.inputs.cfg_fn[-1])
        pdfconfig.backgroundfiles = self.inputs.bkg_fn[-1]
        sqfqgr_path = gp.transform_bkg(pdfconfig, self.iq_data['array'], output_dir=self.inputs.iq_to_gr_path[0], 
                    plot_setting={'marker':'.','color':'green'}, test=True, 
                    gr_fn=gr_fn)    
        gr_data = sqfqgr_path['gr']

        ## Remove headers by reading gr_data into pd.Dataframe and save again
        gr_df = pd.read_csv(gr_data, skiprows=26, names=['r', 'g(r)'], sep =' ')
        gr_df.to_csv(gr_data, index=False, header=False, sep =' ')

        self.gr_data = []
        self.gr_data.append(gr_data)
        self.gr_data.append(gr_df)



    def macro_06_search_and_match(self, gr_fn):
        """macro to search and match the best strucutre, used in kafka consumer
        using package Refinery from updated_pipeline_pdffit2.py  

        Args:
            gr_fn (str): g(r) data path for searching and matching, ex: self.gr_data[0] or self.inputs.gr_fn[0]
                        if using self.gr_data[0], g(r) is generated in workflow
                        if using self.inputs.gr_fn[-1], g(r) is directly read from a file

        Returns:
            str: the file name of the best fitted cif
        """
        # from updated_pipeline_pdffit2 import Refinery
        Refinery = importlib.import_module("updated_pipeline_pdffit2").Refinery
        results_path = self.inputs.results_path[0]
        refinery = Refinery(mystery_path=gr_fn, results_path=results_path, 
                    criteria={"elements":
                        {#["Pb","Se"], 
                        #"$in": ["Cs"], 
                        "$all": ["Pb"],
                        }},
                    strict=[],
                    # strict=["Pb", "S"],
                    pdf_calculator_kwargs={
                        "qmin": 1.0, 
                        "qmax": 18.0,
                        "rmin": 2.0,
                        "rmax": 60.0,
                        "qdamp": 0.031,
                        "qbroad": 0.032
                    },)
        refinery.populate_structures_()
        refinery.populate_pdfs_()
        refinery.apply_metrics_()
        sorted_structures_original = refinery.get_sorted_structures(metric='pearsonr', status='original')
        cif_id = sorted_structures_original[0].material_id
        cif_fn = glob.glob(os.path.join(results_path, f'**{cif_id}**.cif'))[0]
        
        return cif_fn



    def macro_07_fitting_pdf(self, gr_fn, beamline_acronym, 
                            rmax=100.0, qmax=12.0, qdamp=0.031, qbroad=0.032, 
                            fix_APD=True, toler=0.01):
        """macro to do pdf fitting by pdffit2 package, used in kafka consumer

        This macro will
        1. Do pdf refinement of gr_fn
        2. Generate a filename for fitting data by using metadata of stream_name == fluorescence
        3. Save fitting data
        4. Update self.pdf_property
        5. Update fitting data at self.gr_fitting
            self.gr_fitting['R']:       pf.getR()
            self.gr_fitting['pdf_fit']: pf.getpdf_fit()
            self.gr_fitting['array']:   np.array([pf.getR(), pf.getpdf_fit()]) 
            self.gr_fitting['df']:      pd.DataFrame([pf.getR(), pf.getpdf_fit()])

        Args:
            gr_fn (str): g(r) data path for pdf fitting, ex: self.gr_data[0] or self.inputs.gr_fn[0]
                        if using self.gr_data[0], g(r) is generated in workflow
                        if using self.inputs.gr_fn[-1], g(r) is directly read from a file

            beamline_acronym (str): catalog name for tiled to access data
            rmax (float, optional): pdffit2 variable. Defaults to 100.
            qmax (float, optional): pdffit2 variable. Defaults to 12.
            qdamp (float, optional): pdffit2 variable. Defaults to 0.031.
            qbroad (float, optional): pdffit2 variable. Defaults to 0.032.
            fix_APD (bool, optional): pdffit2 variable. Defaults to True.
            toler (float, optional): pdffit2 variable. Defaults to 0.01.
        """

        cif_list = self.inputs.cif_fn[2:]
        pf = pc._pdffit2_CsPbX3(gr_fn, cif_list, rmax=rmax, qmax=qmax, qdamp=qdamp, qbroad=qbroad, 
                                fix_APD=fix_APD, toler=toler, return_pf=True)
        
        phase_fraction = pf.phase_fractions()['mass']
        particel_size = []
        for i in range(pf.num_phases()):
            pf.setphase(i+1)
            particel_size.append(pf.getvar(pf.spdiameter))
        # Grab metadat from stream_name = fluorescence for naming gr file
        fn_uid = de._fn_generator(self.uid, beamline_acronym=beamline_acronym)
        fgr_fn = os.path.join(self.inputs.fitting_pdf_path[0], f'{fn_uid}_scattering.fgr')
        pf.save_pdf(1, f'{fgr_fn}')
        
        self.pdf_property = {}
        self.pdf_property.update({'Br_ratio': phase_fraction[0], 'Br_size':particel_size[0]})
        
        gr_fit_arrary = np.asarray([pf.getR(), pf.getpdf_fit()])
        gr_fit_df = pd.DataFrame()
        gr_fit_df['fit_r'] = pf.getR()
        gr_fit_df['fit_g(r)'] = pf.getpdf_fit()

        self.gr_fitting = {}
        gr_fitting = {  'R':pf.getR(), 
                        'pdf_fit':pf.getpdf_fit(), 
                        'array': gr_fit_arrary, 
                        'df': gr_fit_df}
        self.gr_fitting.update(gr_fitting)

        # self.gr_fitting.append(pf.getR())
        # self.gr_fitting.append(pf.getpdf_fit())
        # self.gr_fitting.append(gr_fit_arrary)
        # self.gr_fitting.append(gr_fit_df)


    def macro_08_no_fitting_pdf(self):
        """macro to update self.gr_fitting while no pdf fitting, used in kafka consumer

        This macro will
        1. Update fitting data at self.gr_fitting
            self.gr_fitting['R']:       []
            self.gr_fitting['pdf_fit']: []
            self.gr_fitting['array']:   None 
            self.gr_fitting['df']:      pd.DataFrame([np.nan, np.nan])
        2. Update gr data at self.gr_data
            self.gr_data[0]: self.inputs.gr_fn[-1]
            self.gr_data[1]: pd.read_csv(gr_fn)

        """
        self.gr_fitting = {}
        gr_fit_arrary = None
        
        gr_fit_df = pd.DataFrame()
        gr_fit_df['fit_r'] = np.nan
        gr_fit_df['fit_g(r)'] = np.nan

        gr_fitting = {  'R':[], 
                        'pdf_fit':[], 
                        'array': gr_fit_arrary, 
                        'df': gr_fit_df}
        self.gr_fitting.update(gr_fitting)
        
        # self.gr_fitting.append([])
        # self.gr_fitting.append([])
        # self.gr_fitting.append(gr_fit_arrary)
        # self.gr_fitting.append(gr_fit_df)
        
        self.pdf_property = {}
        pdf_property={'Br_ratio': np.nan, 'Br_size': np.nan}
        self.pdf_property.update(pdf_property)

        self.gr_data = []
        gr_data = self.inputs.gr_fn[-1]
        gr_df = pd.read_csv(gr_data, skiprows=1, names=['r', 'g(r)'], sep =' ')
        self.gr_data.append(gr_data)
        self.gr_data.append(gr_df)



    def macro_09_qepro_dic(self, stream_name, beamline_acronym):
        """macro to read uv-vis data into dic and save data into .csv file

        This macro will
        1. Read Uv-Vis data according to stream name
        3. Save Uv-Vis data
        4. Update
            self.qepro_dic
            self.metadata_dic
        5. Reset self.PL_goodbad to an empty {}

        Args:
            stream_name (str): the stream name in scan doc to identify scan plan
            beamline_acronym (str): catalog name for tiled to access data
        """
        ## Read data from databroker and turn into dic
        self.qepro_dic, self.metadata_dic = de.read_qepro_by_stream(
                                            self.uid, 
                                            stream_name=stream_name, 
                                            data_agent='tiled', 
                                            beamline_acronym=beamline_acronym)
        self.sample_type = self.metadata_dic['sample_type']
        
        ## Save data in dic into .csv file
        if stream_name == 'take_a_uvvis':
            saving_path = self.inputs.csv_path[1]
        else:
            saving_path = self.inputs.csv_path[0]
        de.dic_to_csv_for_stream(saving_path, self.qepro_dic, self.metadata_dic, stream_name=stream_name)
        print(f'\n** export {stream_name} in uid: {self.uid[0:8]} to ../{os.path.basename(saving_path)} **\n')

        self.PL_goodbad = {}



    def macro_10_good_bad(self, stream_name):
        """macro to identify good/bad data of fluorescence (PL) peak

        This macro will
        1. Identify a good or bad PL peak in 'take_a_uvvis' and 'fluorescence'
        2. Update self.PL_goodbad
            self.PL_goodbad['wavelength']:        wavelenght (nm) of PL
            self.PL_goodbad['percentile_mean']:   intensity of PL (percentile_mean)
            self.PL_goodbad['data_id']:           f'{t0[0]}_{t0[1]}_{metadata_dic["uid"][:8]}'
            self.PL_goodbad['peak']:              peaks from scipy.find_peaks
            self.PL_goodbad['prop']:              properties from scipy.find_peaks

        Args:
            stream_name (str): the stream name in scan doc to identify scan plan
        """
        if self.qepro_dic['QEPro_spectrum_type'][0]==2 and stream_name=='take_a_uvvis':
            print(f'\n*** start to identify good/bad data in stream: {stream_name} ***\n')
            x0, y0, data_id, peak, prop = da._identify_one_in_kafka(
                                            self.qepro_dic, 
                                            self.metadata_dic, 
                                            key_height=self.inputs.key_height[0], 
                                            distance=self.inputs.distance[0], 
                                            height=self.inputs.height[0], 
                                            dummy_test=self.inputs.dummy_kafka[0])

        elif stream_name == 'fluorescence':
            print(f'\n*** start to identify good/bad data in stream: {stream_name} ***\n')
            ## Apply percnetile filtering for PL spectra, defaut percent_range = [30, 100]
            x0, y0, data_id, peak, prop = da._identify_multi_in_kafka(
                                            self.qepro_dic, 
                                            self.metadata_dic, 
                                            key_height=self.inputs.key_height[0], 
                                            distance=self.inputs.distance[0], 
                                            height=self.inputs.height[0], 
                                            dummy_test=self.inputs.dummy_kafka[0], 
                                            percent_range=[40, 100])
            
        self.PL_goodbad = {}
        PL_goodbad = {  'wavelength':np.asarray(x0), 'percentile_mean':np.asarray(y0), 
                        'data_id':data_id, 'peak':peak, 'prop':prop}
        self.PL_goodbad.update(PL_goodbad)



    def macro_11_absorbance(self, stream_name):
        """macro to apply an offset to zero baseline of absorption spectra

        This macro will
        1. Apply a 2D line to fit the baseline of absorption spectra
        2. Update self.abs_data
            self.abs_data['wavelength']:        wavelenght of absorbance nm
            self.abs_data['percentile_mean']:        absorbance array (percentile_mean)    
            self.abs_data['offset']:            absorbance array offset
        3. Update self.abs_fitting
            self.abs_fitting['fit_function']:   da.line_2D
            self.abs_fitting['curve_fit']:      popt_abs (popt from scipy.curve_fit)        

        Args:
            stream_name (str): the stream name in scan doc to identify scan plan
        """

        ## Apply percnetile filtering for absorption spectra, defaut percent_range = [15, 85]
        abs_per = da.percentile_abs(self.qepro_dic['QEPro_x_axis'], 
                                    self.qepro_dic['QEPro_output'], 
                                    percent_range=[10, 70])
        
        print(f'\n*** start to check absorbance at 365b nm in stream: {stream_name} is positive or not***\n')
        # abs_array = qepro_dic['QEPro_output'][1:].mean(axis=0)
        abs_array = abs_per.mean(axis=0)
        wavelength = self.qepro_dic['QEPro_x_axis'][0]

        popt_abs01, _ = da.fit_line_2D(wavelength, abs_array, da.line_2D, x_range=[205, 240], plot=False)
        popt_abs02, _ = da.fit_line_2D(wavelength, abs_array, da.line_2D, x_range=[750, 950], plot=False)
        if abs(popt_abs01[0]) >= abs(popt_abs02[0]):
            popt_abs = popt_abs02
        elif abs(popt_abs01[0]) <= abs(popt_abs02[0]):
            popt_abs = popt_abs01
        
        abs_array_offset = abs_array - da.line_2D(wavelength, *popt_abs)
        print(f'\nFitting function for baseline offset: {da.line_2D}\n')
        ff_abs={'fit_function': da.line_2D, 'curve_fit': popt_abs, 'percentile_mean':abs_array}

        self.abs_data = {}
        self.abs_data.update({'wavelength':wavelength, 'percentile_mean':abs_array, 'offset':abs_array_offset})

        self.abs_fitting = {}
        self.abs_fitting.update(ff_abs)

        de.dic_to_csv_for_stream(self.inputs.csv_path[0], 
                                self.qepro_dic, self.metadata_dic, 
                                stream_name=stream_name, 
                                fitting=ff_abs)


    def macro_12_PL_fitting(self):
        """macro to do peak fitting with gaussian distribution of PL spectra

        This macro will
        1. Apply a 1-peak gaussian fitting to fit the peak of PL spectra
        2. Update self.PL_fitting
            self.PL_fitting['fit_function']:      da._1gauss
            self.PL_fitting['curve_fit']:         popt from scipy.curve_fit    
            self.PL_fitting['fwhm']:              full width at half maximum of highest peak
            self.PL_fitting['peak_emission']:     highest peak position
            self.PL_fitting['wavelength']:        wavelenght (nm) of PL between 400 ~ 800 nm
            self.PL_fitting['intensity']:         intensity of PL (nm) between 400 ~ 800 nm
            self.PL_fitting['shifted_peak_idx']:  peak index between 400 ~ 800 nm
            self.PL_fitting['percentile_mean']:   intensity of PL (percentile_mean)
        """

        x, y, p, f_fit, popt = da._fitting_in_kafka(
                                self.PL_goodbad['wavelength'], 
                                self.PL_goodbad['percentile_mean'], 
                                self.PL_goodbad['data_id'], 
                                self.PL_goodbad['peak'], 
                                self.PL_goodbad['prop'], 
                                dummy_test=self.inputs.dummy_kafka[0])    

        fitted_y = f_fit(x, *popt)
        r2_idx1, _ = da.find_nearest(x, popt[1] - 3*popt[2])
        r2_idx2, _ = da.find_nearest(x, popt[1] + 3*popt[2])
        r_2 = da.r_square(x[r2_idx1:r2_idx2], y[r2_idx1:r2_idx2], fitted_y[r2_idx1:r2_idx2], y_low_limit=0)               

        self.metadata_dic["r_2"] = r_2                                   
        
        if 'gauss' in f_fit.__name__:
            constant = 2.355
        else:
            constant = 1

        intensity_list = []
        peak_list = []
        fwhm_list = []
        for i in range(int(len(popt)/3)):
            intensity_list.append(popt[i*3+0])
            peak_list.append(popt[i*3+1])
            fwhm_list.append(popt[i*3+2]*constant)
        
        peak_emission_id = np.argmax(np.asarray(intensity_list))
        peak_emission = peak_list[peak_emission_id]
        fwhm = fwhm_list[peak_emission_id]
        
        ff={'fit_function': f_fit, 'curve_fit': popt,  
            'fwhm': fwhm, 'peak_emission': peak_emission, 
            'wavelength': x, 'intensity': y, 'shifted_peak_idx': p, 
            'percentile_mean': self.PL_goodbad['percentile_mean']}

        self.PL_fitting = {}
        self.PL_fitting.update(ff)


    def macro_13_PLQY(self):
        """macro to calculate integral of PL peak, PLQY and update optical_property

        This macro will
        1. Integrate PL peak by scipy.integrate.simpson
        2. Calculate PLQY based on the parameters in self.inputs.PLQY
        3. Update self.optical_property
            self.optical_property['PL_integral']:       PL peak integral
            self.optical_property['Absorbance_365']:    absorbance at 365 nm   
            self.optical_property['Peak']:              highest peak position
            self.optical_property['FWHM']:              full width at half maximum of highest peak
            self.optical_property['PLQY']:              Photoluminescence Quantum Yield

        """
        
        ## Integrate PL peak
        x = self.PL_fitting['wavelength']
        y = self.PL_fitting['intensity']
        peak_emission = self.PL_fitting['peak_emission']
        fwhm = self.PL_fitting['fwhm']
        PL_integral_s = integrate.simpson(y)
        
        ## Find absorbance at 365 nm from absorbance stream
        idx1, _ = da.find_nearest(self.abs_data['wavelength'], self.inputs.PLQY[2])
        absorbance_s = self.abs_data['offset'][idx1]

        if self.inputs.PLQY[1] == 'fluorescein':
            plqy = da.plqy_fluorescein(absorbance_s, PL_integral_s, 1.506, *self.inputs.PLQY[3:])
        else:
            plqy = da.plqy_quinine(absorbance_s, PL_integral_s, 1.506, *self.inputs.PLQY[3:])

        
        plqy_dic = {'PL_integral':PL_integral_s, 'Absorbance_365':absorbance_s, 'plqy': plqy}
        self.plqy_dic = {}
        self.plqy_dic.update(plqy_dic)

        optical_property = {'PL_integral':PL_integral_s, 'Absorbance_365':absorbance_s, 
                            'Peak':peak_emission, 'FWHM':fwhm, 'PLQY':plqy}
        self.optical_property = {}
        self.optical_property.update(optical_property)



    def macro_14_upate_agent(self):
        """macro to update agent_data in type of dict for exporting

        This macro will
        1. Update self.agent_data with
            self.optical_property
            self.pdf_property
            self.metadata_dic
            self.abs_fitting
            self.PL_fitting

        """

        ## Creat agent_data in type of dict for exporting
        
        if self.inputs.build_agent[0]:
            if 'agent_target' in self.agent_data.keys():
                pass
            else:
                self.agent_data = {}
        else:
            self.agent_data = {}
        self.agent_data.update(self.optical_property)
        self.agent_data.update(self.pdf_property)
        self.agent_data.update({k:v for k, v in self.metadata_dic.items() if len(np.atleast_1d(v)) == 1})
        self.agent_data.update(de._exprot_rate_agent(self.metadata_dic, self.rate_label_dic, self.agent_data))

        ## Update absorbance offset and fluorescence fitting results inot agent_data
        ff_abs = self.abs_fitting
        ff = self.PL_fitting
        self.agent_data.update({'abs_offset':{'fit_function':ff_abs['fit_function'].__name__, 'popt':ff_abs['curve_fit'].tolist()}})
        self.agent_data.update({'PL_fitting':{'fit_function':ff['fit_function'].__name__, 'popt':ff['curve_fit'].tolist()}})



    def macro_15_save_data(self, stream_name):
        """macro to save processed data and agent data

        This macro will
        1. Save fitting data locally (self.inputs.csv_path[0])
        2. Save processed data in df and agent_data as metadta in sandbox
        3. Save agent_data locally (self.inputs.agent_data_path[0])

        Args:
            stream_name (str): the stream name in scan doc to identify scan plan
        """
        
        ## Save fitting data
        print(f'\nFitting function: {self.PL_fitting["fit_function"]}\n')
        de.dic_to_csv_for_stream(self.inputs.csv_path[0], 
                                self.qepro_dic, 
                                self.metadata_dic, 
                                stream_name = stream_name, 
                                fitting = self.PL_fitting, 
                                plqy_dic = self.plqy_dic)
        print(f'\n** export fitting results complete**\n')

        if stream_name == 'fluorescence':
            self.uid_bundle.append(self.uid)
        
        ## Save processed data in df and agent_data as metadta in sandbox
        if self.inputs.write_to_sandbox[0] and (stream_name == 'fluorescence'):
            df = pd.DataFrame()
            x0 = self.PL_goodbad['wavelength']
            df['wavelength_nm'] = x0
            df['absorbance_mean'] = self.abs_data['percentile_mean']
            df['absorbance_offset'] = self.abs_data['offset']
            df['fluorescence_mean'] = self.PL_goodbad['percentile_mean']
            f_fit = self.PL_fitting['fit_function']
            popt = self.PL_fitting['curve_fit']
            df['fluorescence_fitting'] = f_fit(x0, *popt)

            ## use pd.concat to add various length data together
            try:
                df_new = pd.concat([df, self.iq_data['df'], self.gr_data[1], self.gr_fitting['df']], ignore_index=False, axis=1)
            except (TypeError, KeyError):
                df_new = df

            # entry = sandbox_tiled_client.write_dataframe(df, metadata=agent_data)
            entry = self.sandbox_tiled_client.write_dataframe(df_new, metadata=self.agent_data)
            # uri = sandbox_tiled_client.values()[-1].uri
            uri = entry.uri
            sandbox_uid = uri.split('/')[-1]
            self.uid_sandbox.append(sandbox_uid)
            self.agent_data.update({'sandbox_uid': sandbox_uid})
            print(f"\nwrote to Tiled sandbox uid: {sandbox_uid}")

        ## Save agent_data locally
        if self.inputs.write_agent_data[0] and (stream_name == 'fluorescence'):
            # agent_data.update({'sandbox_uid': sandbox_uid})                               
            with open(f"{self.inputs.agent_data_path[0]}/{self.PL_goodbad['data_id']}.json", "w") as f:
                json.dump(self.agent_data, f, indent=2)

            print(f"\nwrote to {self.inputs.agent_data_path[0]}\n")


    def macro_16_num_good(self, stream_name):
        """macro to add self.PL_goodbad['data_id'] into self.good_data or self.bad_data

        Args:
            stream_name (str): the stream name in scan doc to identify scan plan
        """

        type_peak = type(self.PL_goodbad['peak'])
        type_prop = type(self.PL_goodbad['prop'])

        ## Append good/bad idetified results
        if stream_name == 'take_a_uvvis':

            if (type_peak is np.ndarray) and (type_prop is dict):
                self.good_data.append(self.PL_goodbad['data_id'])

            elif (type(self.PL_goodbad['peak']) == list) and (self.PL_goodbad['prop'] == []):
                self.bad_data.append(self.PL_goodbad['data_id'])
                print(f"\n*** No need to carry out fitting for {stream_name} in uid: {self.uid[:8]} ***\n")
                print(f"\n*** since {stream_name} in uid: {self.uid[:8]} is a bad data.***\n")

        print('\n*** export, identify good/bad, fitting complete ***\n')

        print(f"\n*** {self.sample_type} of uid: {self.uid[:8]} has: ***\n"
                f"*** {self.optical_property = } ***\n"
                f"*** {self.pdf_property = } ***\n")
                

    def macro_17_add_queue(self, stream_name, qserver_process, RM):
        """macro to add queus task to qserver

        This macro will
        1. Add another 'take_a_uvvis' depending on # of good_data
        2. Add new_points preditcted by self.agent

        Args:
            stream_name (str): the stream name in scan doc to identify scan plan
            qserver_process (_LDRD_Kafka.xlsx_to_inputs, optional): qserver parameters read from xlsx.
            RM (REManagerAPI): Run Engine Manager API.
        """

        qin = qserver_process.inputs
        ## Depend on # of good/bad data, add items into queue item or stop 
        if (stream_name == 'take_a_uvvis') and (self.inputs.use_good_bad[0]):     
            if len(self.bad_data) > 3:
                print('*** qsever aborted due to too many bad scans, please check setup ***\n')

                ### Stop all infusing pumps and wash loop
                sq.wash_tube_queue2(qin.pump_list, qin.if_wash, qin.wash_loop, 'ul/min', 
                                zmq_control_addr=qin.zmq_control_addr[0],
                                zmq_info_addr=qin.zmq_info_addr[0])

                RM.queue_stop()
                
            elif (len(self.good_data) <= 2) and (self.inputs.use_good_bad[0]):
                print('*** Add another fluorescence scan to the front of qsever ***\n')
                
                scanplan = BPlan('take_a_uvvis_csv_q', 
                                sample_type=self.metadata_dic['sample_type'], 
                                spectrum_type='Corrected Sample', 
                                correction_type='Dark', 
                                pump_list=qin.pump_list, 
                                precursor_list=qin.precursor_list, 
                                mixer=qin.mixer)
                RM.item_add(scanplan, pos=0)
                
                restplan = BPlan('sleep_sec_q', 2)
                RM.item_add(restplan, pos=0)

                RM.queue_start()

            elif (len(self.good_data) > 2) and (self.inputs.use_good_bad[0]):
                print('*** # of good data is enough so go to the next: bundle plan ***\n')
                self.bad_data.clear()
                self.good_data.clear()
                self.finished.append(self.metadata_dic['sample_type'])
                print(f'After event: good_data = {self.good_data}\n')
                print(f'After event: finished sample = {self.finished}\n')

                RM.queue_start()
        
        ## Add predicted new points from ML agent into qserver
        elif (stream_name == 'fluorescence') and (self.inputs.USE_AGENT_iterate[0]) and (self.continue_iteration[-1]):
            print('*** Add new points from agent to the front of qsever ***\n')
            
            new_points = self.macro_agent(qserver_process, RM, check_target=True)
            
            print(f'*** New points from agent:\n {pprint.pformat(new_points, indent=1)} ***\n')
            
            rate_list = self.auto_rate_list(qin.pump_list, new_points, self.inputs.fix_Br_ratio)

            if self.inputs.post_dilute[0]:
                rate_list.append(sum(rate_list)*self.inputs.post_dilute[1])
    
            qin.sample = de._auto_name_sample(rate_list, prefix=qin.prefix[1:])
            qin.infuse_rates = rate_list
            sq.synthesis_queue_xlsx(qserver_process)

        else:
            print('*** Move to next reaction in Queue ***\n')
            time.sleep(2)
            # RM.queue_start()

        




                
