
import os
import json
import glob
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
# sys.path.insert(0, "/home/xf28id2/src/blop")

from blop import Agent, DOF, Objective


def build_agent(target_correlation=None, agent_data_path='/'):
    # data_path = '/home/xf28id2/data_ZnCl2'
    # agent_data_path = '/home/xf28id2/Documents/ChengHung/agent_data/Cl_02'
    # agent_data_path = agent_data_path
    agent_data_path = '/home/xf28id2/Documents/ChengHung/20250606_XPD/agent_data/agent_data.csv'


    dofs = [
        DOF(description="Cs-rich", name="infusion_rate_Cs-rich", units="uL/min", search_domain=(0, 200)),
        DOF(description="TOABr", name="infusion_rate_Br", units="uL/min", search_domain=(40, 200)),
        DOF(description="Pb-rich", name="infusion_rate_Pb-rich", units="uL/min", search_domain=(0, 200)), 
        DOF(description="20%OLA", name="infusion_rate_OLA", units="uL/min", search_domain=(0, 100)),
    ]
    

    
    # peak_up = peak_target+peak_tolerance
    # peak_down = peak_target-peak_tolerance
    
    # ratio_up = 1-(510-peak_up)*0.99/110
    # ratio_down = 1-(510-peak_down)*0.99/110
    
    objectives = [
        # Objective(description="Peak emission", name="Peak", target=(peak_down, peak_up), weight=100., max_noise=0.25),
        Objective(description="Peak width", name="FWHM", target="min", transform="log", weight=1,max_noise=0.25),
        Objective(description="Quantum yield", name="PLQY", target="max", transform="log", weight=1, max_noise=0.25),
        #Objective(description="CsBr Corre", name="CsBr.gr correlation", target='min', transform="log", weight=1, max_noise=0.25),
        # Objective(description="CsPbBr3 Corre", name="CsPbBr3.gr correlation", target='max', transform="log", weight=1, max_noise=0.1),
        Objective(description="Cs4PbBr6 Corre", name="Cs4PbBr6.gr correlation", target='max', transform="log", weight=5, max_noise=0.1), 
    ]


    print('Start to buid agent')
    agent = Agent(dofs=dofs, objectives=objectives, db=None, verbose=True)
    
    
    fn = agent_data_path
    # names = ['infusion_rate_Cs-rich', 'infusion_rate_Br', 'infusion_rate_Pb-rich', 'infusion_rate_OLA', 
    #          'Peak', 'FWHM', 'PLQY', 
    #          'CsBr.gr correlation', 'CsPbBr3.gr correlation', 'Cs4PbBr6.gr correlation']
    names = ['infusion_rate_Cs-rich', 'infusion_rate_Br', 'infusion_rate_Pb-rich', 'infusion_rate_OLA', 
             'PLQY', 'Peak', 'FWHM', 
             'CsBr.gr correlation', 'Cs4PbBr6.gr correlation', 'CsPbBr3.gr correlation']
    df = pd.read_csv(fn, sep=',', names=names, skiprows=1, index_col=False)
    # print(df)
    # print(agent.dofs.names)
    # print(agent.objectives.names)
    
    for i in range(len(df['PLQY'])):
        x = {k:[df[k][i]] for k in agent.dofs.names}
        y = {k:[df[k][i]] for k in agent.objectives.names}
        metadata = {}
        # metadata = {k:[data.get(k, None)] for k in metadata_keys}
        agent.tell(x=x, y=y, metadata=metadata, train=False, update_models=False)
        
    
    
    agent._construct_all_models()
    agent._train_all_models()

    # print(f'The target of the emission peak is {peak_target} nm.')

    return agent
    
    
    
        # names = ['infusion_rate_Cs-rich', 'infusion_rate_Br', 'infusion_rate_Pb-rich', 'infusion_rate_OLA', 
        #      'PLQY', 'Peak', 'FWHM', 
        #      'CsBr.gr correlation', 'Cs4PbBr6.gr correlation', 'CsPbBr3.gr correlation']
    
    
    
    
    
    # if peak_target > 518:
    #     agent.dofs.infusion_rate_Cl.deactivate()
    #     agent.dofs.infusion_rate_Cl.device.put(0)

    # elif peak_target < 510:        
    #     agent.dofs.infusion_rate_I2.deactivate()
    #     agent.dofs.infusion_rate_I2.device.put(0)
    
    # else:
    #     agent.dofs.infusion_rate_I2.deactivate()
    #     agent.dofs.infusion_rate_I2.device.put(0)
    #     agent.dofs.infusion_rate_Cl.deactivate()
    #     agent.dofs.infusion_rate_Cl.device.put(0)
    
    '''
    metadata_keys = ["time", "uid", "r_2"]
    filepaths = glob.glob(f"{agent_data_path}/*.json")
    filepaths.sort()
    
    # for i in range(len(df['uid'])):
    for fp in tqdm(filepaths):
        with open(fp, "r") as f:
            data = json.load(f)
        # print(data)
        # data = {}
        # for key in df.keys():
        #     data[key] = df[key][i]
            
        r_2_min = 0.70
        try: 
            if data['r_2'] < r_2_min:
                print(f'Skip because "r_2" of {data["uid"]} is {data["r_2"]:.2f} < {r_2_min}.')
            else: 
                x = {k:[data[k]] for k in agent.dofs.names}
                y = {k:[data[k]] for k in agent.objectives.names}
                metadata = {k:[data.get(k, None)] for k in metadata_keys}
                agent.tell(x=x, y=y, metadata=metadata, train=False, update_models=False)
        
        except (KeyError):
            print(f'{os.path.basename(fp)} has no "r_2".')


    agent._construct_all_models()
    agent._train_all_models()

    print(f'The target of the emission peak is {peak_target} nm.')

    return agent

    # print(agent.ask("qei", n=1))
    # print(agent.ask("qr", n=36))
    
    '''


'''
agent.posterior
import torch

res = agent.ask('qem', n=1)
agent.posterior(torch.tensor(res['points'])).mean

x = torch.tensor([[ 24.0426776 , 159.30614932, 101.20516362]])
agent.posterior(x)
agent.posterior(x).mean
agent.plot_acquisition(); plt.show()


18/2: agent
18/3: agent.table
18/4: agent.table.Peak
18/5: plt.rcParams['font.size'] = 4
18/6: import matplotlib.pyplot as plt
18/7: plt.rcParams['font.size'] = 4
18/8: agent.plot_objectives(); plt.show()
18/9: agent.objectives
18/10: agent.ask("qem", n=1)
18/11: agent.ask("qei", n=1)
18/12: agent.ask("qei", n=1)
18/13: import torch
18/14: x = torch.tensor(res[0])
18/15: res = agent.ask("qem", n=4)
18/16: x = torch.tensor(res[0])
18/17: agent.posterior(x).mean
18/18: agent.best
18/19: agent.objectives
18/20: post = agent.posterior(x)
18/21: post.mean
18/22: post.sigma
18/23: post.stddev
18/24: agent.objectives
18/25: agent.plot_acquisition(); plt.show()
18/26: agent.plot_constraint(); plt.show()
18/27: agent.dofs
18/28: agent.objectives
18/29: agent.best

 '''