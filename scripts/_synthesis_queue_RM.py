import numpy as np
# import pandas as pd
# from bluesky_queueserver.manager.comms import zmq_single_request
import _data_export as de
from bluesky_queueserver_api.zmq import REManagerAPI
from bluesky_queueserver_api import BPlan, BInst
from ophyd.sim import det, noisy_det
# from _LDRD_Kafka import xlsx_to_inputs

## Pass qsever parameters by xlsx_to_inputs
## Arrange tasks of for PQDs synthesis
def synthesis_queue_xlsx(parameter_obj):
	"""
	Pass qsever parameters by xlsx_to_inputs
	Arrange tasks of for PQDs synthesis

	Args:
		parameter_obj (xlsx_to_inputs): parameters passing to qserver 
		(example: pump_list = parameter_obj.inputs.pump_list)
	"""
	
	qsp = parameter_obj.inputs

	syringe_list = qsp.syringe_list
	pump_list = qsp.pump_list
	auto_set_target_list = qsp.auto_set_target_list[0]
	set_target_list = qsp.set_target_list
	target_vol_list = qsp.target_vol_list
	rate_list = qsp.infuse_rates
	syringe_mater_list = qsp.syringe_mater_list
	precursor_list = qsp.precursor_list
	mixer = qsp.mixer
	resident_t_ratio = qsp.resident_t_ratio 
	prefix = qsp.prefix
	sample = qsp.sample
	wait_dilute = qsp.wait_dilute
	if_wash = qsp.if_wash
	wash_loop = qsp.wash_loop
	# wash_sapphire = qsp.wash_sapphire
	rate_unit = qsp.rate_unit[0]
	name_by_prefix = qsp.name_by_prefix[0]
	det2 = qsp.uvvis_config[0]
	num_abs = qsp.uvvis_config[1]
	num_flu = qsp.uvvis_config[2]
	det1 = qsp.perkin_config[0]
	det1_frame_rate = qsp.perkin_config[1]
	det1_time = qsp.perkin_config[2]
	pos = qsp.pos[0]
	dummy_qserver = qsp.dummy_qserver[0]
	is_iteration = qsp.is_iteration[0]
	zmq_control_addr = qsp.zmq_control_addr[0]
	zmq_info_addr = qsp.zmq_info_addr[0]


	RM = REManagerAPI(zmq_control_addr=zmq_control_addr, zmq_info_addr=zmq_info_addr)

	if name_by_prefix:
		sample = de._auto_name_sample(rate_list, prefix=prefix)
	                                                                                 
	rate_list = np.asarray(rate_list, dtype=np.float32)
	if len(rate_list.shape) == 1:
		rate_list = rate_list.reshape(1, rate_list.shape[0])
		rate_list = rate_list.tolist()
	else:
		rate_list = rate_list.tolist()

	if auto_set_target_list:
		set_target_list = np.zeros([len(rate_list), len(pump_list)]).tolist()
	
	else:
		set_target_list = np.asarray(set_target_list, dtype=np.int8)
		if len(set_target_list.shape) == 1:
			set_target_list = set_target_list.reshape(1, set_target_list.shape[0])
			set_target_list = set_target_list.tolist()
		else:
			set_target_list = set_target_list.tolist()
		
	num_pumps = int(len(wash_loop)/3)
	# 0. stop infuese for all pumps
	flowplan = BPlan('stop_group', pump_list + [wash_loop[1+i*3] for i in range(num_pumps)])
	RM.item_add(flowplan, pos=pos)
	

	for i in range(len(rate_list)):
		# for i in range(2): 
		## 1. Set i infuese rates
		rate_list2 = rate_list[i][:-1] + [60]
		for sl, pl, ir, tvl, stl, sml in zip(
											syringe_list, 
											pump_list, 
											rate_list2, 
											target_vol_list, 
											set_target_list[i], 
											syringe_mater_list
											):
			
			# ir = float(ir)
			# stl = int(stl)
			flowplan = BPlan('set_group_infuse2', [sl], [pl],
							rate_list = [ir], 
							target_vol_list = [tvl], 
							set_target_list = [stl], 
							syringe_mater_list = [sml], 
							rate_unit = rate_unit)
			RM.item_add(flowplan, pos=pos)


		# ## 2. Start infuese
		# if precursor_list[-1] == 'Toluene':
		# 	flowplan = BPlan('start_group_infuse', pump_list[:-1], rate_list[i][:-1])
		
		# else:
		# 	flowplan = BPlan('start_group_infuse', pump_list, rate_list[i])
		
		flowplan = BPlan('start_group_infuse', pump_list, rate_list[i])
		RM.item_add(flowplan, pos=pos)


		## 3. Wait for equilibrium
		if len(mixer) == 1:
			if (precursor_list[-1] == 'Toluene') and (precursor_list[-2] == 'PF_oil'):
				mixer_pump_list = [[mixer[0], *pump_list[:-2]]]
			# if (precursor_list[-1] == 'Toluene') and ('CsPb' in precursor_list[0]):
			# 	mixer_pump_list = [[mixer[0], *pump_list[:2]]]
			# elif (len(precursor_list)==6) and (precursor_list[-1] == 'Toluene') and ('OLA' in precursor_list[3]):
			# 	mixer_pump_list = [[mixer[0], *pump_list[:4]]]
			# elif (len(precursor_list)==5) and (precursor_list[-1] == 'Toluene') and ('Cs-rich' in precursor_list[0]):
			# 	mixer_pump_list = [[mixer[0], *pump_list[:3]]]
			else:
				mixer_pump_list = [[mixer[0], *pump_list]]
		elif len(mixer) == 2:
			if precursor_list[-1] == 'Toluene':
				mixer_pump_list = [[mixer[0], *pump_list[:2]], [mixer[1], *pump_list[:-1]]]
			else:
				mixer_pump_list = [[mixer[0], *pump_list[:2]], [mixer[1], *pump_list]]
		
		if dummy_qserver:
			restplan = BPlan('sleep_sec_q', qsp.dummy_qserver[1])
			RM.item_add(restplan, pos=pos)
		
		else:
			if is_iteration:
				rest_time = resident_t_ratio[-1]
			
			elif len(resident_t_ratio) == 1:
				rest_time = resident_t_ratio[0]
			elif len(resident_t_ratio) > 1 and i==0:
				rest_time = resident_t_ratio[0]
			elif len(resident_t_ratio) > 1 and i>0:
				rest_time = resident_t_ratio[-1]

			restplan = BPlan('wait_equilibrium2', mixer_pump_list, ratio=rest_time)
			RM.item_add(restplan, pos=pos)


		## 3.1 Wait for 30 secpnds for post dilute
		if precursor_list[-1] == 'Toluene':
			# flowplan = BPlan('start_group_infuse', [pump_list[-1]], [rate_list[i][-1]])
			flowplan = BPlan('set_group_infuse2', [syringe_list[-1]], [pump_list[-1]],
							rate_list = [rate_list[i][-1]], 
							target_vol_list = [target_vol_list[-1]], 
							set_target_list = [set_target_list[i][-1]], 
							syringe_mater_list = [syringe_mater_list[-1]], 
							rate_unit = rate_unit)
			RM.item_add(flowplan, pos=pos)
			
			restplan = BPlan('sleep_sec_q', qsp.wait_dilute[0])
			RM.item_add(restplan, pos=pos)
   
		
  
  		# ## 4.0 Configure area detector in Qserver
		# if det1 == 'pe1c' or det1 == 'pe2c':
		# 	scanplan = BPlan('configure_area_det', 
		# 					det=det1, 
		# 					exposure=1, 
		# 					acq_time=det1_frame_rate)
		# 	RM.item_add(scanplan, pos=pos)
		

		## 4-1. Take a fluorescence peak to check reaction
		scanplan = BPlan('take_a_uvvis_csv_q', sample_type=sample[i], 
						spectrum_type='Corrected Sample', 
                        correction_type='Dark', 
						pump_list=pump_list, 
						precursor_list=precursor_list, 
                        mixer=mixer)
		RM.item_add(scanplan, pos=pos)
    

		# ## 4-2. Take a Absorption spectra to check reaction
        # scanplan = BPlan('take_a_uvvis_csv_q', sample_type=sample[i], 
		# 				spectrum_type='Absorbtion', 
        #                 correction_type='Reference', 
		# 				pump_list=pump_list, 
		# 				precursor_list=precursor_list, 
        #                 mixer=mixer)
        # RM.item_add(scanplan, pos=pos)


		#### Kafka check data here.

		## 5. Sleep for 5 seconds for Kafak to check good/bad data
		restplan = BPlan('sleep_sec_q', 2)
		RM.item_add(restplan, pos=pos)
		

		# ## 6.0 Print global parameters in Qserver
		# scanplan = BPlan('print_glbl_qserver')
		# RM.item_add(scanplan, pos=pos)
  
		if det1 == 'pe1c' or det1 == 'pe2c':
			## 6.1 Configure area detector in Qserver
			scanplan = BPlan('configure_area_det', 
							det=det1, 
							exposure=det1_time, 
							acq_time=det1_frame_rate)
			RM.item_add(scanplan, pos=pos)


		## 6. Start xray_uvvis bundle plan to take real data  ('pe1c' or 'det')
		scanplan = BPlan('xray_uvvis_plan2', det1, det2, 
						num_abs=num_abs, 
						num_flu=num_flu, 
						sample_type=sample[i], 
						spectrum_type='Absorbtion', 
						correction_type='Reference', 
						pump_list=pump_list, 
						precursor_list=precursor_list, 
						mixer=mixer)
		RM.item_add(scanplan, pos=pos)
        
        ## 6.1 sleep 20 seconds for stopping
		restplan = BPlan('sleep_sec_q', qsp.wait_dilute[1])
		RM.item_add(restplan, pos=pos)
        

		######  Kafka analyze data here. #######

		## 7. Wash the loop and mixer
		if if_wash[0] == 1:
			wash_tube_queue2(pump_list, if_wash, wash_loop, rate_unit, 
							pos=[pos,pos,pos,pos,pos], 
							zmq_control_addr=zmq_control_addr,
							zmq_info_addr=zmq_info_addr)
		elif wash_tube[0] == 0:
			inst1 = BInst("queue_stop")
			RM.item_add(inst1, pos='front')


	# 8. stop infuese for all pumps
	flowplan = BPlan('stop_group', pump_list)
	RM.item_add(flowplan, pos=pos)




## Arrange tasks of for PQDs synthesis
def synthesis_queue(
                    syringe_list, 
                    pump_list,
					auto_set_target_list, 
                    set_target_list, 
                    target_vol_list, 
                    rate_list, 
                    syringe_mater_list, 
                    precursor_list,
                    mixer, 
                    resident_t_ratio, 
                    prefix, 
                    sample, 
                    wash_tube, 
					rate_unit='ul/min',
					name_by_prefix=True,  
					num_abs=5, 
					num_flu=5, 
					det1 = det, 
					det1_time=15, 
		            det1_frame_rate=0.2,
					pos='back',
                    dummy_qserver=False,
					is_iteration=False, 
					zmq_control_addr='tcp://localhost:60615', 
					zmq_info_addr='tcp://localhost:60625', 
                    ):

	RM = REManagerAPI(zmq_control_addr=zmq_control_addr, zmq_info_addr=zmq_info_addr)

	if name_by_prefix:
		sample = de._auto_name_sample(rate_list, prefix=prefix)
	                                                                                 
	rate_list = np.asarray(rate_list, dtype=np.float32)
	if len(rate_list.shape) == 1:
		rate_list = rate_list.reshape(1, rate_list.shape[0])
		rate_list = rate_list.tolist()
	else:
		rate_list = rate_list.tolist()

	
	if auto_set_target_list[0]:
		set_target_list = np.zeros([len(rate_list), len(pump_list)]).tolist()	
	
	else:
		set_target_list = np.asarray(set_target_list, dtype=np.int8)
		if len(set_target_list.shape) == 1:
			set_target_list = set_target_list.reshape(1, set_target_list.shape[0])
			set_target_list = set_target_list.tolist()
		else:
			set_target_list = set_target_list.tolist()
		

	# 0. stop infuese for all pumps
	flowplan = BPlan('stop_group', pump_list + [wash_tube[2], wash_tube[5]])
	RM.item_add(flowplan, pos=pos)
	
	
	for i in range(len(rate_list)):
		# for i in range(2): 
		## 1. Set i infuese rates
		for sl, pl, ir, tvl, stl, sml in zip(
											syringe_list, 
											pump_list, 
											rate_list[i], 
											target_vol_list, 
											set_target_list[i], 
											syringe_mater_list
											):
			
			# ir = float(ir)
			# stl = int(stl)

			flowplan = BPlan('set_group_infuse2', [sl], [pl],
							rate_list = [ir], 
							target_vol_list = [tvl], 
							set_target_list = [stl], 
							syringe_mater_list = [sml], 
							rate_unit = rate_unit)
			RM.item_add(flowplan, pos=pos)


		## 2. Start infuese
		if precursor_list[-1] == 'Toluene':
			flowplan = BPlan('start_group_infuse', pump_list[:-1], rate_list[i][:-1])
		
		else:
			flowplan = BPlan('start_group_infuse', pump_list, rate_list[i])
		
		RM.item_add(flowplan, pos=pos)


		## 3. Wait for equilibrium
		if len(mixer) == 1:
			if precursor_list[-1] == 'Toluene':
				mixer_pump_list = [[mixer[0], *pump_list[:2]]]
			else:
				mixer_pump_list = [[mixer[0], *pump_list]]
		elif len(mixer) == 2:
			if precursor_list[-1] == 'Toluene':
				mixer_pump_list = [[mixer[0], *pump_list[:2]], [mixer[1], *pump_list[:-1]]]
			else:
				mixer_pump_list = [[mixer[0], *pump_list[:2]], [mixer[1], *pump_list]]
		
		if dummy_qserver:
			restplan = BPlan('sleep_sec_q', 5)
			RM.item_add(restplan, pos=pos)
		
		else:
			if is_iteration:
				rest_time = resident_t_ratio[-1]
			
			elif len(resident_t_ratio) == 1:
				rest_time = resident_t_ratio[0]
			elif len(resident_t_ratio) > 1 and i==0:
				rest_time = resident_t_ratio[0]
			elif len(resident_t_ratio) > 1 and i>0:
				rest_time = resident_t_ratio[-1]

			restplan = BPlan('wait_equilibrium2', mixer_pump_list, ratio=rest_time)
			RM.item_add(restplan, pos=pos)


		## 3.1 Wait for 30 secpnds for post dilute
		if precursor_list[-1] == 'Toluene':
			flowplan = BPlan('start_group_infuse', [pump_list[-1]], [rate_list[i][-1]])
			RM.item_add(flowplan, pos=pos)
			
			restplan = BPlan('sleep_sec_q', 30)
			RM.item_add(restplan, pos=pos)
   
		
  
  		## 4.0 Configure area detector in Qserver
		if det1 == 'pe1c':
			scanplan = BPlan('configure_area_det', 
							det='pe1c', 
							exposure=1, 
							acq_time=det1_frame_rate)
			RM.item_add(scanplan, pos=pos)
		

		## 4-1. Take a fluorescence peak to check reaction
		scanplan = BPlan('take_a_uvvis_csv_q', sample_type=sample[i], 
						spectrum_type='Corrected Sample', 
                        correction_type='Dark', 
						pump_list=pump_list, 
						precursor_list=precursor_list, 
                        mixer=mixer)
		RM.item_add(scanplan, pos=pos)
    

		# ## 4-2. Take a Absorption spectra to check reaction
        # scanplan = BPlan('take_a_uvvis_csv_q', sample_type=sample[i], 
		# 				spectrum_type='Absorbtion', 
        #                 correction_type='Reference', 
		# 				pump_list=pump_list, 
		# 				precursor_list=precursor_list, 
        #                 mixer=mixer)
        # RM.item_add(scanplan, pos=pos)


		#### Kafka check data here.

		## 5. Sleep for 5 seconds for Kafak to check good/bad data
		restplan = BPlan('sleep_sec_q', 5)
		RM.item_add(restplan, pos=pos)
		

		# ## 6.0 Print global parameters in Qserver
		# scanplan = BPlan('print_glbl_qserver')
		# RM.item_add(scanplan, pos=pos)
  
		if det1 == 'pe1c':
			## 6.1 Configure area detector in Qserver
			scanplan = BPlan('configure_area_det', 
							det=det1, 
							exposure=det1_time, 
							acq_time=det1_frame_rate)
			RM.item_add(scanplan, pos=pos)


		## 6. Start xray_uvvis bundle plan to take real data  ('pe1c' or 'det')
		scanplan = BPlan('xray_uvvis_plan2', det1, 'qepro', 
						num_abs=num_abs, 
						num_flu=num_flu, 
						sample_type=sample[i], 
						spectrum_type='Absorbtion', 
						correction_type='Reference', 
						pump_list=pump_list, 
						precursor_list=precursor_list, 
						mixer=mixer)
		RM.item_add(scanplan, pos=pos)
        
        ## 6.1 sleep 20 seconds for stopping
		restplan = BPlan('sleep_sec_q', 20)
		RM.item_add(restplan, pos=pos)
        

		######  Kafka analyze data here. #######

		## 7. Wash the loop and mixer
		if wash_tube[0] == 0:
			wash_tube_queue2(pump_list, wash_tube, rate_unit, 
							pos=[pos,pos,pos,pos,pos], 
							zmq_control_addr=zmq_control_addr,
							zmq_info_addr=zmq_info_addr)
		elif wash_tube[0] == 1:
			inst1 = BInst("queue_stop")
			RM.item_add(inst1, pos='front')


	# 8. stop infuese for all pumps
	flowplan = BPlan('stop_group', pump_list)
	RM.item_add(flowplan, pos=pos)




## Arrange tasks of for video of PQDs synthesis
def synthesis_queue3(
                    syringe_list, 
                    pump_list, 
                    set_target_list, 
                    target_vol_list, 
                    rate_list, 
                    syringe_mater_list, 
                    precursor_list,
                    mixer, 
                    resident_t_ratio, 
                    prefix, 
                    sample, 
                    wash_tube, 
					rate_unit='ul/min',
					name_by_prefix=True,  
					num_abs=5, 
					num_flu=5, 
					det1 = det, 
					det1_time=15, 
		            det1_frame_rate=0.2,
					pos='back',
                    dummy_qserver=False,
					is_iteration=False, 
					zmq_control_addr='tcp://localhost:60615', 
					zmq_info_addr='tcp://localhost:60625', 
                    ):

	RM = REManagerAPI(zmq_control_addr=zmq_control_addr, zmq_info_addr=zmq_info_addr)

	if name_by_prefix:
		sample = de._auto_name_sample(rate_list, prefix=prefix)
	                                                                                 
	rate_list = np.asarray(rate_list, dtype=np.float32)
	if len(rate_list.shape) == 1:
		rate_list = rate_list.reshape(1, rate_list.shape[0])
		rate_list = rate_list.tolist()
	else:
		rate_list = rate_list.tolist()

	set_target_list = np.asarray(set_target_list, dtype=np.int8)
	if len(set_target_list.shape) == 1:
		set_target_list = set_target_list.reshape(1, set_target_list.shape[0])
		set_target_list = set_target_list.tolist()
	else:
		set_target_list = set_target_list.tolist()
		

	# 0. stop infuese for all pumps
	flowplan = BPlan('stop_group', pump_list + [wash_tube[2], wash_tube[5]])
	RM.item_add(flowplan, pos=pos)
	
	
	for i in range(len(rate_list)):
		# for i in range(2): 
		## 1. Set i infuese rates
		for sl, pl, ir, tvl, stl, sml in zip(
											syringe_list, 
											pump_list, 
											rate_list[i], 
											target_vol_list, 
											set_target_list[i], 
											syringe_mater_list
											):
			
			# ir = float(ir)
			# stl = int(stl)

			flowplan = BPlan('set_group_infuse2', [sl], [pl],
							rate_list = [ir], 
							target_vol_list = [tvl], 
							set_target_list = [stl], 
							syringe_mater_list = [sml], 
							rate_unit = rate_unit)
			RM.item_add(flowplan, pos=pos)


		## 2. Start infuese
		if precursor_list[-1] == 'Toluene':
			flowplan = BPlan('start_group_infuse', pump_list[:-1], rate_list[i][:-1])
		
		else:
			flowplan = BPlan('start_group_infuse', pump_list, rate_list[i])
		
		RM.item_add(flowplan, pos=pos)


		## 3. Wait for equilibrium
		if len(mixer) == 1:
			if precursor_list[-1] == 'Toluene':
				mixer_pump_list = [[mixer[0], *pump_list[:2]]]
			else:
				mixer_pump_list = [[mixer[0], *pump_list]]
		elif len(mixer) == 2:
			if precursor_list[-1] == 'Toluene':
				mixer_pump_list = [[mixer[0], *pump_list[:2]], [mixer[1], *pump_list[:-1]]]
			else:
				mixer_pump_list = [[mixer[0], *pump_list[:2]], [mixer[1], *pump_list]]
		
		if dummy_qserver:
			restplan = BPlan('sleep_sec_q', 5)
			RM.item_add(restplan, pos=pos)
		
		else:
			if is_iteration:
				rest_time = resident_t_ratio[-1]
			
			elif len(resident_t_ratio) == 1:
				rest_time = resident_t_ratio[0]
			elif len(resident_t_ratio) > 1 and i==0:
				rest_time = resident_t_ratio[0]
			elif len(resident_t_ratio) > 1 and i>0:
				rest_time = resident_t_ratio[-1]

			restplan = BPlan('wait_equilibrium2', mixer_pump_list, ratio=rest_time)
			RM.item_add(restplan, pos=pos)


		# ## 3.1 Wait for 30 secpnds for post dilute
		# if precursor_list[-1] == 'Toluene':
		# 	flowplan = BPlan('start_group_infuse', [pump_list[-1]], [rate_list[i][-1]])
		# 	RM.item_add(flowplan, pos=pos)
			
		# 	restplan = BPlan('sleep_sec_q', 30)
		# 	RM.item_add(restplan, pos=pos)
   
		
  
  		## 4.0 Configure area detector in Qserver
		if det1 == 'pe1c':
			scanplan = BPlan('configure_area_det', 
							det='pe1c', 
							exposure=1, 
							acq_time=det1_frame_rate)
			RM.item_add(scanplan, pos=pos)
		

		## 4-1. Take a fluorescence peak to check reaction
		scanplan = BPlan('take_a_uvvis_csv_q3', sample_type=sample[i], 
						spectrum_type='Corrected Sample', 
                        correction_type='Dark', 
						pump_list=pump_list, 
						precursor_list=precursor_list, 
                        mixer=mixer)
		RM.item_add(scanplan, pos=pos)
    

		# ## 4-2. Take a Absorption spectra to check reaction
        # scanplan = BPlan('take_a_uvvis_csv_q', sample_type=sample[i], 
		# 				spectrum_type='Absorbtion', 
        #                 correction_type='Reference', 
		# 				pump_list=pump_list, 
		# 				precursor_list=precursor_list, 
        #                 mixer=mixer)
        # RM.item_add(scanplan, pos=pos)


		#### Kafka check data here.

		## 5. Sleep for 5 seconds for Kafak to check good/bad data
		restplan = BPlan('sleep_sec_q', 5)
		RM.item_add(restplan, pos=pos)
		

		# ## 6.0 Print global parameters in Qserver
		# scanplan = BPlan('print_glbl_qserver')
		# RM.item_add(scanplan, pos=pos)
  
		if det1 == 'pe1c':
			## 6.1 Configure area detector in Qserver
			scanplan = BPlan('configure_area_det', 
							det=det1, 
							exposure=det1_time, 
							acq_time=det1_frame_rate)
			RM.item_add(scanplan, pos=pos)


		## 6. Start xray_uvvis bundle plan to take real data  ('pe1c' or 'det')
		scanplan = BPlan('xray_uvvis_plan3', det1, 'qepro', 
						num_abs=num_abs, 
						num_flu=num_flu, 
						sample_type=sample[i], 
						spectrum_type='Absorbtion', 
						correction_type='Reference', 
						pump_list=pump_list, 
						precursor_list=precursor_list, 
						mixer=mixer)
		RM.item_add(scanplan, pos=pos)
        
        ## 6.1 sleep 20 seconds for stopping
		restplan = BPlan('sleep_sec_q', 20)
		RM.item_add(restplan, pos=pos)
        

		######  Kafka analyze data here. #######

		## 7. Wash the loop and mixer
		if wash_tube[0] == 0:
			wash_tube_queue3(pump_list, wash_tube, rate_unit, 
							pos=[pos,pos,pos,pos,pos], 
							zmq_control_addr=zmq_control_addr,
							zmq_info_addr=zmq_info_addr)
		elif wash_tube[0] == 1:
			inst1 = BInst("queue_stop")
			RM.item_add(inst1, pos='front')


	# 8. stop infuese for all pumps
	flowplan = BPlan('stop_group', pump_list)
	RM.item_add(flowplan, pos=pos)






## wash loop with one solvent
def wash_tube_queue(pump_list, wash_tube, rate_unit, 
					pos=[0,1,2,3,4], 
					zmq_control_addr='tcp://localhost:60615', 
					zmq_info_addr='tcp://localhost:60625'):

	RM = REManagerAPI(zmq_control_addr=zmq_control_addr, zmq_info_addr=zmq_info_addr)

	### Stop all infusing pumps
	flowplan = BPlan('stop_group', pump_list)
	RM.item_add(flowplan, pos=pos[0])


	### Set up washing tube/loop
	flowplan = BPlan('set_group_infuse2', [wash_tube[1]], [wash_tube[2]], 
					rate_list=[wash_tube[3]], 
					target_vol_list=['30 ml'], 
					set_target_list=[False], 
					syringe_mater_list = ['steel'], 
					rate_unit=rate_unit)
	RM.item_add(flowplan, pos=pos[1])	
	
	
	### Start washing tube/loop
	flowplan = BPlan('start_group_infuse', [wash_tube[2]], [wash_tube[3]])
	RM.item_add(flowplan, pos=pos[2])	


	### Wash loop/tube for xxx seconds
	restplan = BPlan('sleep_sec_q', wash_tube[4])
	RM.item_add(restplan, pos=pos[3])	
	


	### Stop washing
	flowplan = BPlan('stop_group', [wash_tube[2]])
	RM.item_add(flowplan, pos=pos[4])



## wash loop with two solvents
def wash_tube_queue2(pump_list, if_wash, wash_loop, rate_unit, 
					pos=[0,1,2,3,4], 
					zmq_control_addr='tcp://localhost:60615', 
					zmq_info_addr='tcp://localhost:60625'):

	RM = REManagerAPI(zmq_control_addr=zmq_control_addr, zmq_info_addr=zmq_info_addr)

	### Stop all infusing pumps
	flowplan = BPlan('stop_group', pump_list)
	RM.item_add(flowplan, pos=pos[0])

	num_pumps = int(len(wash_loop)/3)
	### Set up washing tube/loop
	flowplan = BPlan('set_group_infuse2', [wash_loop[0+i*3] for i in range(num_pumps)], [wash_loop[1+i*3] for i in range(num_pumps)], 
					rate_list=[wash_loop[2+i*3] for i in range(num_pumps)], 
					target_vol_list=['30 ml', '15 ml'], 
					set_target_list=[False, False], 
					syringe_mater_list = ['steel', 'plastic_BD'], 
					rate_unit=rate_unit)
	RM.item_add(flowplan, pos=pos[1])	
	
	
	### Start washing tube/loop
	flowplan = BPlan('start_group_infuse', [wash_loop[1+i*3] for i in range(num_pumps)], [wash_loop[2+i*3] for i in range(num_pumps)])
	RM.item_add(flowplan, pos=pos[2])	


	### Wash loop/tube for xxx seconds
	restplan = BPlan('sleep_sec_q', if_wash[1])
	RM.item_add(restplan, pos=pos[3])	
	


	### Stop washing
	flowplan = BPlan('stop_group', [wash_loop[1+i*3] for i in range(num_pumps)])
	RM.item_add(flowplan, pos=pos[4])





## dummy wash loop with two solvents for video
def wash_tube_queue3(pump_list, wash_tube, rate_unit, 
					pos=[0,1,2,3,4], 
					zmq_control_addr='tcp://localhost:60615', 
					zmq_info_addr='tcp://localhost:60625'):

	RM = REManagerAPI(zmq_control_addr=zmq_control_addr, zmq_info_addr=zmq_info_addr)

	### Stop all infusing pumps
	flowplan = BPlan('stop_group', pump_list)
	RM.item_add(flowplan, pos=pos[0])


	### Set up washing tube/loop
	flowplan = BPlan('set_group_infuse2', [wash_tube[1], wash_tube[4]], [wash_tube[2], wash_tube[5]], 
					rate_list=[wash_tube[3], wash_tube[6]], 
					target_vol_list=['30 ml', '15 ml'], 
					set_target_list=[False, False], 
					syringe_mater_list = ['steel', 'steel'], 
					rate_unit=rate_unit)
	RM.item_add(flowplan, pos=pos[1])	
	
	
	### Start washing tube/loop
	flowplan = BPlan('start_group_infuse', [wash_tube[2], wash_tube[5]], [wash_tube[3], wash_tube[6]])
	RM.item_add(flowplan, pos=pos[2])	


	### Wash loop/tube for xxx seconds
	restplan = BPlan('sleep_sec_q', wash_tube[7])
	RM.item_add(restplan, pos=pos[3])	
	


	### Stop washing
	flowplan = BPlan('stop_group', [wash_tube[2], wash_tube[5]])
	RM.item_add(flowplan, pos=pos[4])





## Auto generate sample name with given prefix and infuse_rate
## If prefix = None, 'Pre00', 'Pre01', 'Pre02', ... will be used.
# def _auto_name_sample(infuse_rates, prefix=None):
#     infuse_rates = np.asarray(infuse_rates)

#     if len(infuse_rates.shape) == 1:
#         infuse_rates = infuse_rates.reshape(1, infuse_rates.shape[0])

#     if prefix == None:
#         prefix_list = [f'Pre{i:02d}' for i in range(infuse_rates.shape[1])]
#     else:
#         prefix_list = prefix

#     sample = []
#     for i in range(infuse_rates.shape[0]):
#         name = ''
#         for j in range(infuse_rates.shape[1]):
#             int_rate = int(round(float(infuse_rates[i][j]), 0))
#             name += f'{prefix_list[j]}_{int_rate:03d}_'
#         sample.append(name[:-1])
    
#     return sample





'''
 'plans_allowed': {'count': '{...}',
                   'scan': '{...}',
                   'LED_on': '{...}',
                   'LED_off': '{...}',
                   'shutter_open': '{...}',
                   'shutter_close': '{...}',
                   'deuterium_on': '{...}',
                   'deuterium_off': '{...}',
                   'halogen_on': '{...}',
                   'halogen_off': '{...}',
                   'reset_pumps2': '{...}',
                   'set_group_infuse': '{...}',
                   'set_group_withdraw': '{...}',
                   'start_group_infuse': '{...}',
                   'start_group_withdraw': '{...}',
                   'stop_group': '{...}',
                   'insitu_test': '{...}',
                   'setup_collection_q': '{...}',
                   'take_ref_bkg_q': '{...}',
                   'take_a_uvvis_csv_q': '{...}',
                   'sleep_sec_q': '{...}',
                   'wait_equilibrium': '{...}',
                   'xray_uvvis_plan': '{...}'},


## Add ietration of ML agent into Qsever
def iterate_queue(
				new_points, 
                syringe_list, 
                pump_list, 
                target_vol_list, 
                syringe_mater_list, 
                precursor_list,
                mixer, 
                resident_t_ratio, 
                prefix, 
                wash_tube, 
				name_by_prefix=True, 
				num_abs=5,
				num_flu=5,
                new_points_unit='ul/min', 
				pos='back', 
                dummy_qserver=False, 
                ):
	        
	# To do: unit conversion
	# if rate_unit != 'ul/min':
	
	# num_iteration = new_points[0].shape[0]
	set_target_list = [0 for i in range(new_points[0].shape[1])]
	sample = _auto_name_sample(new_points[0], prefix=prefix)

	# infuse_rates_float = new_points[0]
	# unit_array = np.full([infuse_rates_float.shape[0], infuse_rates_float.shape[1]],' ul/min', dtype='U7')
	# infuse_rates_string = infuse_rates_float.astype('U25')
	# infuse_rates = np.char.add(infuse_rates_string, unit_array)
	

	return synthesis_queue(
						syringe_list=syringe_list, 
						pump_list=pump_list, 
						set_target_list=set_target_list, 
						target_vol_list=target_vol_list, 
						rate_list=new_points[0], 
						syringe_mater_list=syringe_mater_list, 
						precursor_list=precursor_list,
						mixer=mixer, 
						resident_t_ratio=resident_t_ratio, 
						prefix=prefix, 
						sample=sample, 
						wash_tube=wash_tube, 
						name_by_prefix=name_by_prefix, 
						num_abs=num_abs,
						num_flu=num_flu,
						rate_unit=new_points_unit, 
						pos='back', 
						dummy_qserver=False, 
						)

'''


