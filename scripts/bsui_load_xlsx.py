import importlib
## Add comments for all the modules
# LK = importlib.import_module("_LDRD_Kafka")
# sq = importlib.import_module("_synthesis_queue_RM")
de = importlib.import_module("_data_export")

# ophyd_map = {
# 	'dds2_p1': dds2_p1, 
#  	'dds2_p2': dds2_p2, 
#   	'dds3_p2': dds3_p2, 
#    	'dds3_p1': dds3_p1, 
# 	'dds1_p1': dds1_p1, 
#  	'ultra1': ultra1, 
#   	'ultra2': ultra2, 
#    	'dds1_p2': dds1_p2, 
#     'qepro': qepro, 
#     'pe1c': pe1c
# }

def _qserver_inputs():
    """Define namespace for Qserver inputs

    Returns:
        list: list for Qserver inputs
    All the namesapce in this list should be found in the excel spreadsheet.
    """
    qserver_list=[
            'zmq_control_addr', 'zmq_info_addr', 'http_server_uri', 
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




# xlsx_fn = '/home/xf28id2/Documents/ChengHung/inputs_qserver_kafka_v2.xlsx'
# xlsx_fn = '/home/xf28id2/.ipython/profile_collection/scripts/inputs_qserver_kafka_v2.xlsx'
# xlsx_fn = '/home/xf28id2/.ipython/profile_collection_ldrd20-31/scripts/inputs_qserver_kafka_v2_test.xlsx'
xlsx_fn = '/home/xf28id2/.ipython/profile_collection_ldrd20-31/scripts/inputs_qserver_kafka_v2.xlsx'

## Input varaibales for Qserver, reading from xlsx_fn by given sheet name
# qserver_process = LK.xlsx_to_inputs(LK._qserver_inputs(), xlsx_fn=xlsx_fn, sheet_name='qserver_XPD_test')
qserver_process = xlsx_to_inputs(_qserver_inputs(), xlsx_fn=xlsx_fn, sheet_name='qserver_XPD')
qin = qserver_process.inputs


