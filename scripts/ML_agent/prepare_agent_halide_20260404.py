import os
import json
import glob
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
# sys.path.insert(0, "/home/xf28id2/src/blop")

from ax.api.protocols import IMetric
from blop import Agent, RangeDOF, Objective, ScalarizedObjective, OutcomeConstraint

from scripts.ML_agent.evaluation_halide import HalideEvaluation


def build_agent(
    peak_target=660, peak_tolerance=5, size_target=6, agent_data_path="/", use_OAm=False
):
    # data_path = '/home/xf28id2/data_ZnCl2'
    # agent_data_path = '/home/xf28id2/Documents/ChengHung/agent_data/Cl_02'
    # agent_data_path = agent_data_path
    # agent_data_path = '/nsls2/data/xpd-new/legacy/processed/LDRD_chl/20240312_dataset_800/agent_data_update_quinine_20240312.csv'

    if use_OAm:
        dofs = [
            RangeDOF(
                name="infusion_rate_CsPb",
                bounds=(20, 80),
                parameter_type="float",
            ),
            # DOF(description="TOABr", name="infusion_rate_Br", units="uL/min", search_domain=(50, 200)),
            # DOF(description="ZnI2", name="infusion_rate_I2", units="uL/min", search_domain=(10, 190)),
            RangeDOF(
                name="infusion_rate_Cl",
                bounds=(10, 190),
                parameter_type="float",
            ),
            RangeDOF(
                name="infusion_rate_OAm",
                bounds=(0, 70),
                parameter_type="float",
            ),
        ]

    else:
        dofs = [
            RangeDOF(
                name="infusion_rate_CsPb",
                bounds=(10, 200),
                parameter_type="float",
            ),
            RangeDOF(
                name="infusion_rate_Br",
                bounds=(5, 200),
                parameter_type="float",
            ),
            RangeDOF(
                name="infusion_rate_I2",
                bounds=(0, 200),
                parameter_type="float",
            ),
            # DOF(description="ZnCl2", name="infusion_rate_Cl", units="uL/min", search_domain=(0, 200)),
        ]

    peak_up = peak_target + peak_tolerance
    peak_down = peak_target - peak_tolerance

    # ratio_up = 1-(510-peak_up)*0.99/110
    # ratio_down = 1-(510-peak_down)*0.99/110

    # Separate tracking metric that we have constraints on
    #   - Don't minimize or maximize peak
    #   - Simply get it in-bounds
    peak_metric = IMetric(name="Peak")
    peak_constraints = [
        OutcomeConstraint(f"p >= {peak_down}", p=peak_metric),
        OutcomeConstraint(f"p <= {peak_up}", p=peak_metric),
    ]

    # TODO: If desired, we can use scalarization
    # scalarized_objective = ScalarizedObjective(
    #     "-50 * x + 100 * y",
    #     minimize=False,
    #     x="log_FWHM",
    #     y="log_PLQY",
    # )

    objectives = [
        # Objective(description="Peak emission", name="Peak", target=peak_target, transform="log",weight=10., max_noise=0.25),
        Objective(
            name="log_FWHM",
            minimize=True,
            # transform="log",
            # weight=50.0,
            # max_noise=0.25, # TODO: How critical is max_noise? Requires custom BoTorch model in new Blop
        ),
        Objective(
            name="log_PLQY",
            minimize=False,
            # transform="log",
            # weight=100.0,
            # max_noise=0.25, # TODO: How critical is max_noise? Requires custom BoTorch model in new Blop
        ),
        # Objective(description="Particle size", name="Br_size", target=(size_target-1.5, size_target+1.5), transform="log", weight=0.1, max_noise=0.25),
        # Objective(description="Phase ratio", name="Br_ratio", target=(ratio_down, ratio_up), transform="log", weight=0.1, max_noise=0.25),
    ]

    # objectives = [
    #     Objective(name="Peak emission", key="peak_emission", target=525, units="nm"),
    #     Objective(name="Peafilepath

    print("Start to build agent")
    agent = Agent(
        sensors=[],
        dofs=dofs,
        objectives=objectives,
        evaluation_function=evaluation_function,
        outcome_constraints=peak_constraints,
    )

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

    metadata_keys = ["time", "uid", "r_2"]

    # filepaths = glob.glob(f"{agent_data_path}/*.json")
    # filepaths.sort()

    # fn = agent_data_path + '/' + 'agent_data_update_quinine_CsPbCl3.csv'
    fn = agent_data_path

    names = [
        "infusion_rate_CsPb",
        "infusion_rate_Br",
        "infusion_rate_I2",
        "infusion_rate_Cl",
        "Peak",
        "FWHM",
        "PLQY",
        "time",
        "uid",
        "r_2",
    ]

    df = pd.read_csv(fn, sep=" ", names=names, skiprows=1, index_col=False)

    for i in range(len(df["uid"])):
        # for fp in tqdm(filepaths):
        #     with open(fp, "r") as f:
        #         data = json.load(f)
        # print(data)
        data = {}
        for key in df.keys():
            data[key] = df[key][i]

        r_2_min = 0.70
        try:
            if data["r_2"] < r_2_min:
                print(
                    f'Skip because "r_2" of {data["uid"]} is {data["r_2"]:.2f} < {r_2_min}.'
                )
            else:
                x = {k: [data[k]] for k in agent.dofs.names}
                y = {k: [data[k]] for k in agent.objectives.names}
                # metadata = {}
                metadata = {k: [data.get(k, None)] for k in metadata_keys}
                # print(f'\n{x = }')
                # print(f'{y = }\n')
                agent.tell(
                    x=x, y=y, metadata=metadata, train=False, update_models=False
                )

        except KeyError:
            print(f'{os.path.basename(fp)} has no "r_2".')

    agent._construct_all_models()
    agent._train_all_models()

    print(f"The target of the emission peak is {peak_target} nm.")

    return agent

    # print(agent.ask("qei", n=1))
    # print(agent.ask("qr", n=36))


"""
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

 """

