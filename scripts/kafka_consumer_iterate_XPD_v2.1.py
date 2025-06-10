import os
import datetime
import pprint
import uuid
# from bluesky_kafka import RemoteDispatcher
from bluesky_kafka.consume import BasicConsumer
import matplotlib.pyplot as plt
import numpy as np
from tiled.client import from_uri, from_profile

# Set limit of numbers of open files
import resource
resource.setrlimit(resource.RLIMIT_NOFILE, (65536, 65536))

import importlib
## Add comments for all the modules
LK = importlib.import_module("_LDRD_Kafka")
sq = importlib.import_module("_synthesis_queue_RM")
de = importlib.import_module("_data_export")
# da = importlib.import_module("_data_analysis")
plot_uvvis = importlib.import_module("_plot_helper").plot_uvvis

from bluesky_queueserver_api.zmq import REManagerAPI
# from bluesky_queueserver_api import BPlan, BInst

try:
    from nslsii import _read_bluesky_kafka_config_file  # nslsii <0.7.0
except (ImportError, AttributeError):
    from nslsii.kafka_utils import _read_bluesky_kafka_config_file  # nslsii >=0.7.0

# these two lines allow a stale plot to remain interactive and prevent
# the current plot from stealing focus.  thanks to Tom:
# https://nsls2.slack.com/archives/C02D9V72QH1/p1674589090772499
plt.ion()
plt.rcParams["figure.raise_window"] = False

# xlsx_fn = '/home/xf28id2/Documents/ChengHung/inputs_qserver_kafka_v2.xlsx'
# xlsx_fn = '/home/xf28id2/.ipython/profile_collection/scripts/inputs_qserver_kafka_v2.xlsx'
# xlsx_fn = '/home/xf28id2/.ipython/profile_collection_ldrd20-31/scripts/inputs_qserver_kafka_v2_test.xlsx'
xlsx_fn = '/home/xf28id2/.ipython/profile_collection_ldrd20-31/scripts/inputs_qserver_kafka_v2.xlsx'

## Input varaibales for Qserver, reading from xlsx_fn by given sheet name
# qserver_process = LK.xlsx_to_inputs(LK._qserver_inputs(), xlsx_fn=xlsx_fn, sheet_name='qserver_XPD_test')
qserver_process = LK.xlsx_to_inputs(LK._qserver_inputs(), xlsx_fn=xlsx_fn, sheet_name='qserver_XPD')
qin = qserver_process.inputs

## Input varaibales for Kafka, reading from xlsx_fn by given sheet name
# kafka_process = LK.xlsx_to_inputs(LK._kafka_inputs(), xlsx_fn=xlsx_fn, sheet_name='kafka_process_test', is_kafka=True)
kafka_process = LK.xlsx_to_inputs(LK._kafka_inputs(), xlsx_fn=xlsx_fn, sheet_name='kafka_process', is_kafka=True)
kin = kafka_process.inputs

## Define RE Manager API as RM 
RM = REManagerAPI(zmq_control_addr=qin.zmq_control_addr[0], zmq_info_addr=qin.zmq_info_addr[0])

## Make the first prediction from kafka_process.agent
if kin.use_1st_prediction[0]:
    first_points = kafka_process.macro_agent(qserver_process, RM, check_target=False, is_1st=True)
    rate_list = kafka_process.auto_rate_list(qin.pump_list, first_points, kin.fix_Br_ratio)
    if kin.post_dilute[0]:
        sum_active = sum(rate_list)
        rate_list.append(sum_active/9)   ## append PF rate
        rate_list.append(sum_active*kin.post_dilute[1])  ## append toluene rate
    qin.infuse_rates = rate_list
    print(f'\n{rate_list = }\n')

## Import Qserver parameters to RE Manager
sq.synthesis_queue_xlsx(qserver_process)

## Auto name samples by prefix
if qin.name_by_prefix[0]:
    sample = de._auto_name_sample(qin.infuse_rates, prefix=qin.prefix)
print(f'Sample: {sample}')


def print_kafka_messages(beamline_acronym_01, 
                        beamline_acronym_02, 
                        kafka_process=kafka_process, 
                        qserver_process=qserver_process, 
                        RM=RM, ):

    
    """Print kafka message from beamline_acronym

    Args:
        beamline_acronym_01 (str): subscribed topics for raw data publishing (ex: xpd, xpd-ldrd20-31)
        beamline_acronym_02 (str): subscribed topics for analysis data publishing (ex: xpd-analysis)
        kafka_process (_LDRD_Kafka.xlsx_to_inputs, optional): kafka parameters read from xlsx. Defaults to kafka_process.
        qserver_process (_LDRD_Kafka.xlsx_to_inputs, optional): qserver parameters read from xlsx. Defaults to qserver_process.
        RM (REManagerAPI, optional): Run Engine Manager API. Defaults to RM.
    """

    kin = kafka_process.inputs
    qin = qserver_process.inputs

    print(f"Listening for Kafka messages for\n"
                                            f"     raw data: {beamline_acronym_01}\n"
                                            f"analysis data: {beamline_acronym_02}")
    print(f'Defaul parameters:\n'
          f'                  csv path: {kin.csv_path[0]}\n'
          f'                  key height: {kin.key_height[0]}\n'
          f'                  height: {kin.height[0]}\n'
          f'                  distance: {kin.distance[0]}\n'
          
          f'{bool(kin.use_good_bad[0]) = }\n'
          f'{bool(kin.post_dilute[0]) = }\n'
          f'{bool(kin.write_agent_data[0]) = }\n'
          f'{kin.agent_data_path[0] = }\n'

          f'{bool(kin.USE_AGENT_iterate[0]) = }\n'
          f'{kin.peak_target[0] = } nm\n'

          f'{bool(kin.iq_to_gr[0]) = }\n'
          f'{kin.iq_to_gr_path[0] = }\n'

          f'{bool(kin.search_and_match[0]) = }\n'

          f'{bool(kin.fitting_pdf[0]) = }\n'
          f'{kin.fitting_pdf_path[0] = }\n'

          f'{bool(kin.write_to_sandbox[0]) = }\n'
          f'{qin.zmq_control_addr[0] = }')

    ## Assignt raw data tiled clients
    kin.beamline_acronym.append(beamline_acronym_01)
    kafka_process.tiled_client = from_profile(beamline_acronym_01)
    ## 'xpd-analysis' is not a catalog name so can't be accessed in databroker

    ## Append good/bad data folder to csv_path
    kin.csv_path.append(os.path.join(kin.csv_path[0], 'good_bad'))

    ## Make directory for good/bad data folder
    try:
        os.mkdir(kin.csv_path[1])
    except FileExistsError:
        pass
    

    def print_message(consumer, doctype, doc):
        name, message = doc
        # print(f"contents: {pprint.pformat(message)}\n")
        
        ## macro_00: print metadata when doc name is start and reset self.uid to an empty list
        ######### While document (name == 'start') and ('topic' in message) #########
        ##         Only print metadata when the docuemnt is from pdfstream         ##
        ##                      reset self.uid to an empty list                    ##
        #############################################################################
        if (name == 'start') and ('topic' in message):
            print(
                f"\n\n{datetime.datetime.now().isoformat()} documents {name}\n"
                f"document keys: {list(message.keys())}")
            
            kafka_process.macro_00_print_start(message)


        ## macro_01: stop queue and get uid for take_a_uvvis scan
        #########  (name == 'stop') and ('take_a_uvvis' in message['num_events'])  ##########
        ##     Only take a Uv-Vis, no X-ray data but still do analysis of pdfstream        ##
        ##                   Stop queue first for identify good/bad data                   ##
        ##                   Obtain raw data uid in bluesky doc, message                   ##
        #####################################################################################         
        elif (name == 'stop') and ('take_a_uvvis' in message['num_events']):
            print('\n*** qsever stop for data export, identification, and fitting ***\n')

            print(f"\n\n{datetime.datetime.now().isoformat()} documents {name}\n"
                  f"contents: {pprint.pformat(message)}")
            
            kafka_process.macro_01_stop_queue_uid(RM, message)

            
            
        ## macro_02: get iq from sandbox
        ######### While document (name == 'event') and ('topic' in message) ##########
        ##        key 'topic' is added into the doc of xpd-analysis in pdfstream    ##
        ##        Read uid of analysis data from message['data']['chi_I']           ##
        ##        Get I(Q) data from the integral of 2D image by pdfstream          ##
        ##############################################################################
        elif (name == 'event') and ('topic' in message):
            # print(f"\n\n\n{datetime.datetime.now().isoformat()} documents {name}\n"
            #       f"contents: {pprint.pformat(message)}")

            iq_I_uid  = message['data']['chi_I']
            kafka_process.macro_02_get_iq(iq_I_uid)

        

        ## macro_03: get raw data uid from metadata of sandbox doc
        #### (name == 'stop') and ('topic' in message) and ('primary' in message['num_events']) ####
        ##         With taking xray_uvvis_plan and analysis of pdfstream finished                 ##
        ##       Sleep 1 second and assign uid, stream_list from kafka_process.entry[-1]          ##
        ##          No need to stop queue since the net queue task is wahsing loop                ##
        ############################################################################################
        elif (name == 'stop') and ('topic' in message) and ('primary' in message['num_events']):
            print(f"\n\n\n{datetime.datetime.now().isoformat()} documents {name}\n"
                  f"contents: {pprint.pformat(message)}"
            )
            kafka_process.macro_03_get_uid()


        ##############  (name == 'stop') and uid_is_str and check_event_name   ##############
        ##  Move to data fitting, calculation when uid is assigned, its type is a string,  ##
        ##   and stream name is 'primary':      for xray_uvvis_plan2 after pdfstream       ##
        ##    or stream name is 'take_a_uvvis': for take_a_uvvis (raw data)                ##
        ##################################################################################### 
        uid_is_str = (type(kafka_process.uid) is str)
        try:
            check_event_name = ('primary' in message['num_events'] or 'take_a_uvvis' in message['num_events'])
        except KeyError:
            check_event_name = False
        
        if (name == 'stop') and uid_is_str and check_event_name:
            print(f'\n**** start to export uid: {kafka_process.uid} ****\n')
            print(f'\n**** with stream name in {kafka_process.stream_list} ****\n')


            ## macro_04 ~ macro_07 or 08
            ####################  'scattering' in kafka_process.stream_list   ###################
            ##    Process X-ray scattering data (PDF): iq to gr, search & match, pdf fitting   ##
            ##              obtain phase fraction & particle size from g(r)                    ##
            ##################################################################################### 
            if 'scattering' in kafka_process.stream_list:
                # Get metadata from stream_name fluorescence for plotting
                kafka_process.qepro_dic, kafka_process.metadata_dic = de.read_qepro_by_stream(
                    kafka_process.uid, stream_name='fluorescence', data_agent='tiled', 
                    beamline_acronym=beamline_acronym_01)
                u = plot_uvvis(kafka_process.qepro_dic, kafka_process.metadata_dic)

                ## macro_04: setting dummy pdf data for test, e.g., CsPbBr2
                if kin.dummy_pdf[0]:
                    kafka_process.macro_04_dummy_pdf()

                ## macro_05: do i(q) to g(r) through pdfstream
                if kin.iq_to_gr[0]:
                    kafka_process.macro_05_iq_to_gr(beamline_acronym_01)

                ## macro_06: do search and match
                if kin.search_and_match[0]:
                    # cif_fn = kafka_process.macro_06_search_and_match(kin.gr_fn[0])
                    cif_fn = kafka_process.macro_06_search_and_match(kafka_process.gr_data[0])                    
                    print(f'\n\n*** After matching, the most correlated strucuture is\n' 
                          f'*** {cif_fn} ***\n\n')
                    
                ## macro_061: calculate pearson correlation of gr data vs. simulated
                if kin.pearson_pdf[0]:
                    # pearson_results = kafka_process.macro_061_pearson_pdf(kin.gr_fn[0])
                    pearson_results = kafka_process.macro_061_pearson_pdf(kafka_process.gr_data[0])                    
                    print(f'\n\n*** After Pearson calculation, the correlations to simulation are\n')
                    pprint.pprint(pearson_results)
                    print('\n\n')
                
                ## macro_07: do pdf fitting and update kafka_process.gr_fitting
                if kin.fitting_pdf[0]:
                    kafka_process.macro_07_fitting_pdf(
                        kafka_process.gr_data[0], beamline_acronym_01, 
                        rmax=100.0, qmax=12.0, qdamp=0.031, qbroad=0.032, 
                        fix_APD=True, toler=0.01
                        )
                ## macro_08: not do pdf fitting but also update kafka_process.gr_fitting
                else:
                    kafka_process.macro_08_no_fitting_pdf()
                
                if kin.iq_to_gr[0]:
                    u.plot_iq_to_gr(kafka_process.iq_data['array'], kafka_process.gr_data[1].to_numpy().T, gr_fit=kafka_process.gr_fitting['array'])
                
                ## remove 'scattering' from stream_list to avoid redundant work in next for loop
                kafka_process.stream_list.remove('scattering')
            

            ########  Other stream names except 'scattering' in kafka_process.stream_list   #####
            ##   Process Uv-Vis data: Export, plotting, fitting, calculate # of good/bad data  ## 
            ##                        add queue item, PLQY                                     ##
            ##   Agent: save agent data locally & to sandbox, make optimization                ##
            ##################################################################################### 
            for stream_name in kafka_process.stream_list:

                ##############################  macro_09 ~ macro_11   ###############################
                ##              read uv-vis data into dic and save data into .csv file             ##
                ##      Idenfify good/bad data if it is a fluorescence scan in 'take_a_uvvis'      ##
                ##       Avergae fluorescence scans in 'fluorescence' and idenfify good/bad        ##
                ##################################################################################### 
                
                ## macro_09: read uv-vis data into dic and save data into .csv file
                kafka_process.macro_09_qepro_dic(stream_name, beamline_acronym_01)
                
                ## Plot data in dic
                u = plot_uvvis(kafka_process.qepro_dic, kafka_process.metadata_dic)
                if len(kafka_process.good_data)==0 and len(kafka_process.bad_data)==0:
                    clear_fig=True
                else:
                    clear_fig=False
                u.plot_data(clear_fig=clear_fig)
                print(f'\n** Plot {stream_name} in uid: {kafka_process.uid[0:8]} complete **\n')
                

                ## macro_10_good_bad: Idenfify good/bad data if it is a fluorescence scan in 'take_a_uvvis'
                if kafka_process.qepro_dic['QEPro_spectrum_type'][0]==2 and stream_name=='take_a_uvvis':
                    print(f'\n*** start to identify good/bad data in stream: {stream_name} ***\n')
                    kafka_process.macro_10_good_bad(stream_name)
                
                
                ## macro_11_absorbance: Apply an offset to zero baseline of absorption spectra
                elif stream_name == 'absorbance':
                    print(f'\n*** start to filter absorbance within 15%-85% due to PF oil phase***\n')
                    kafka_process.macro_11_absorbance(stream_name)

                    u.plot_offfset(kafka_process.abs_data['wavelength'], kafka_process.abs_fitting['fit_function'], kafka_process.abs_fitting['curve_fit'])
                    print(f'\n** export offset results of absorption spectra complete**\n')

                
                ## macro_10_good_bad: Avergae scans in 'fluorescence' and idenfify good/bad
                elif stream_name == 'fluorescence':
                    print(f'\n*** start to identify good/bad data in stream: {stream_name} ***\n')
                    kafka_process.macro_10_good_bad(stream_name)
                    
                    type_peak = type(kafka_process.PL_goodbad['peak'])
                    type_prop = type(kafka_process.PL_goodbad['prop'])
                    
                    if (type_peak is np.ndarray) and (type_prop is dict):
                        label_uid = f'{kafka_process.uid[0:8]}_{kafka_process.metadata_dic["sample_type"]}'
                        u.plot_average_good(kafka_process.PL_goodbad['wavelength'], 
                                            kafka_process.PL_goodbad['percentile_mean'], 
                                            label=label_uid, clf_limit=9)

 
                ## Skip peak fitting if qepro type is absorbance
                if kafka_process.qepro_dic['QEPro_spectrum_type'][0] == 3:  
                    print(f"\n*** No need to carry out fitting for {stream_name} in uid: {kafka_process.uid[:8]} ***\n")

                ## macro_12 ~ macro_16
                else: 
                    ##############  (type(peak) is np.ndarray) and (type(prop) is dict)  ################
                    ##   for a good data, type(peak) will be a np.array and type(prop) will be a dict  ##
                    ##                fit the good data, export/plotting fitting results               ##
                    ##          append data_id into good_data or bad_data for calculate numbers        ##
                    ##################################################################################### 
                    type_peak = type(kafka_process.PL_goodbad['peak'])
                    type_prop = type(kafka_process.PL_goodbad['prop'])

                    ## When PL data is good
                    if (type_peak is np.ndarray) and (type_prop is dict):
                        
                        ## macro_12_PL_fitting: do peak fitting with gaussian distribution of PL spectra
                        kafka_process.macro_12_PL_fitting()

                        ## Calculate PLQY for fluorescence stream
                        if (stream_name == 'fluorescence') and (kin.PLQY[0]==1):
                            
                            ## macro_13_PLQY: calculate integral of PL peak, PLQY and update optical_property
                            kafka_process.macro_13_PLQY()
                            label_uid = f'{kafka_process.uid[0:8]}_{kafka_process.metadata_dic["sample_type"]}'
                            # u.plot_CsPbX3(kafka_process.PL_fitting['wavelength'], kafka_process.PL_fitting['intensity'], 
                            #             kafka_process.PL_fitting['peak_emission'], label=label_uid, clf_limit=9)

                            ## macro_14_agent_data: Creat agent_data in type of dict for exporting
                            kafka_process.macro_14_upate_agent()

                        else:
                            kafka_process.plqy_dic ={}
                            kafka_process.optical_property = {}
                        
                        ## Plot fitting data
                        u.plot_peak_fit(kafka_process.PL_fitting['wavelength'], 
                                        kafka_process.PL_fitting['intensity'], 
                                        kafka_process.PL_fitting['fit_function'], 
                                        kafka_process.PL_fitting['curve_fit'], 
                                        peak=kafka_process.PL_fitting['shifted_peak_idx'], 
                                        fill_between=True)
                        print(f'\n** plot fitting results complete**\n')
                        
                        ## macro_15_save_data: Save processed data and agent data
                        kafka_process.macro_15_save_data(stream_name)
                    
                    ## Bad data: No peak fitting
                    elif  (type_peak is list) and (stream_name == 'fluorescence'):
                        kafka_process.macro_12_PL_fitting(has_peak=False)
                        kafka_process.macro_13_PLQY(has_peak=False)
                        kafka_process.macro_14_upate_agent(has_peak=False)
                                                    
                        ## macro_15_save_data: Save processed data and agent data
                        kafka_process.macro_15_save_data(stream_name, has_peak=False)
                        
                    ##  macro_16_num_good: Add self.PL_goodbad['data_id'] into self.good_data or self.bad_data
                    kafka_process.macro_16_num_good(stream_name)


            print(f'*** Accumulated num of good data: {len(kafka_process.good_data)} ***\n')
            print(f'{kafka_process.good_data = }\n')
            print(f'*** Accumulated num of bad data: {len(kafka_process.bad_data)} ***\n')
            print('########### Events printing division ############\n')
            

            #################  macro_17_add_queue: Add queus task to qserver  ###################
            ##        Depend on # of good/bad data, add items into queue item or stop          ##
            ##                     Add nother 'take_a_uvvis' into queue                        ##
            ##          Make prediction by self.agent and add new_points to queue              ##
            ##################################################################################### 
            kafka_process.macro_17_add_queue(stream_name, qserver_process, RM)

        
        if (name == 'stop') and ('fluorescence' in kafka_process.stream_list):
            kafka_process.save_kafka_dict('/home/xf28id2/Documents/ChengHung/kafka_dict_log')
            
        
        ## Reset kafka_process.uid to an empty list for event doc identification 
        kafka_process.uid = []
        kafka_process.stream_list = []

    
    kafka_config = _read_bluesky_kafka_config_file(config_file_path="/etc/bluesky/kafka.yml")

    # this consumer should not be in a group with other consumers
    #   so generate a unique consumer group id for it
    unique_group_id = f"echo-{beamline_acronym_01}-{str(uuid.uuid4())[:8]}"

    kafka_consumer = BasicConsumer(
        topics=[f"{beamline_acronym_01}.bluesky.runengine.documents", 
                f"{beamline_acronym_02}.bluesky.runengine.documents"],
        bootstrap_servers=kafka_config["bootstrap_servers"],
        group_id=unique_group_id,
        consumer_config=kafka_config["runengine_producer_config"],
        process_message = print_message,
    )

    try:
        kafka_consumer.start_polling(work_during_wait=lambda : plt.pause(.1))
    except KeyboardInterrupt:
        print('\nExiting Kafka consumer')
        return()


if __name__ == "__main__":
    import sys
    print_kafka_messages(sys.argv[1], sys.argv[2])
