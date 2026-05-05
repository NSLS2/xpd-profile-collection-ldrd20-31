# Make ophyd listen to pyepics.
import nslsii
import ophyd.signal

import matplotlib.pyplot as plt

ophyd.signal.EpicsSignal.set_defaults(connection_timeout=5)
# See docstring for nslsii.configure_base() for more details
# this command takes away much of the boilerplate for setting up a profile
# (such as setting up best effort callbacks etc)
# nslsii.configure_base(get_ipython().user_ns, 'xpd', pbar=True, bec=True,
#                       magics=True, mpl=True, epics_context=False)


# At the end of every run, verify that files were saved and
# print a confirmation message.
from bluesky.callbacks.broker import verify_files_saved, post_run
# RE.subscribe(post_run(verify_files_saved, db), 'stop')
from bluesky import RunEngine

from databroker import Broker
from bluesky.callbacks.best_effort import BestEffortCallback


# Uncomment the following lines to turn on verbose messages for
# debugging.
# import logging
# ophyd.logger.setLevel(logging.DEBUG)
# logging.basicConfig(level=loggisng.DEBUG)

# nslsii.configure_base(get_ipython().user_ns, "xpd",
#                       publish_documents_with_kafka=True)

# nslsii.configure_base(get_ipython().user_ns, "xpd-ldrd20-31",
#                       publish_documents_with_kafka=True)

RE = RunEngine({})

# db = Broker.named("xpd-ldrd20-31")
# db = Broker.named("xpd")
# db = Broker.named("xpd-analysis")
# bec = BestEffortCallback()

# RE.subscribe(db.insert)
# RE.subscribe(bec)
# res = nslsii.configure_kafka_publisher(RE, beamline_name="xpd-ldrd20-31")
# res = nslsii.configure_kafka_publisher(RE, beamline_name="xpd")
# res = nslsii.configure_kafka_publisher(RE, beamline_name="xpd-analysis")



RE.md['facility'] = 'NSLS-II'
RE.md['group'] = 'XPD'
RE.md['beamline_id'] = '28-ID-2'

import subprocess


from ophyd import Device, EpicsMotor, EpicsSignal, EpicsSignalRO
from ophyd import Component as Cpt
import time
import pandas as pd
import numpy as np
from ophyd.sim import det4, noisy_det, motor  # simulated detector, motor
import h5py
from datetime import datetime
import glob
import bluesky.plan_stubs as bps
from bluesky.plans import count, scan


# Make plots update live while scans run.
# from bluesky.utils import install_kicker
# install_kicker()
#%matplotlib notebook
#from bluesky.utils import install_nb_kicker
#install_nb_kicker()


# ## Prepare Data Storage
# from databroker import Broker
# db = Broker.named('xpd-ldrd20-31')

# from databroker import catalog
# db = catalog['xpd']


# ## Add a Progress Bar
# from bluesky.utils import ProgressBarManager
# RE.waiting_hook = ProgressBarManager()

def show_env():
    # this is not guaranteed to work as you can start IPython without hacking
    # the path via activate
    proc = subprocess.Popen(["conda", "list"], stdout=subprocess.PIPE)
    out, err = proc.communicate()
    a = out.decode('utf-8')
    b = a.split('\n')
    print(b[0].split('/')[-1][:-1])


from bluesky_queueserver import is_re_worker_active
if is_re_worker_active():
    print('<code without interactive features, e.g. reading data from a file>')

    # Publish run documents to the bluesky-0MQ-proxy so subscribers
    # (RemoteDispatcher, TiledWriter, etc.) can receive them.
    import os
    _zmq_proxy_in = os.environ.get("BLUESKY_ZMQ_PROXY_IN_ADDR", "")
    if _zmq_proxy_in:
        from bluesky.callbacks.zmq import Publisher as ZmqPublisher
        # Parse "tcp://localhost:5577" into ("localhost", 5577) tuple
        _host_port = _zmq_proxy_in.replace("tcp://", "").split(":")
        _zmq_addr_tuple = (_host_port[0], int(_host_port[1]))
        _publisher = ZmqPublisher(_zmq_addr_tuple)
        RE.subscribe(_publisher)
        print(f"[STARTUP] RE subscribed ZMQ Publisher -> {_zmq_addr_tuple}")
else:
    print('<code with interactive features, e.g. manual data input>')