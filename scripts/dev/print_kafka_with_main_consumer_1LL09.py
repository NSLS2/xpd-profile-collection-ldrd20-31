import os
import datetime
import pprint
import uuid
# from bluesky_kafka import RemoteDispatcher
from bluesky_kafka.consume import BasicConsumer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import databroker
from tiled.client import from_profile

# import _data_export as de
# from _plot_helper import plot_uvvis
# import _data_analysis as da

# db = databroker.Broker.named('xpd-ldrd20-31')
# catalog = databroker.catalog['xpd-ldrd20-31']

# global db, catalog
# db = databroker.Broker.named(beamline_acronym_01)
# catalog = databroker.catalog[f'{beamline_acronym_01}']

try:
    from nslsii import _read_bluesky_kafka_config_file  # nslsii <0.7.0
except (ImportError, AttributeError):
    from nslsii.kafka_utils import _read_bluesky_kafka_config_file  # nslsii >=0.7.0

# these two lines allow a stale plot to remain interactive and prevent
# the current plot from stealing focus.  thanks to Tom:
# https://nsls2.slack.com/archives/C02D9V72QH1/p1674589090772499
plt.ion()
plt.rcParams["figure.raise_window"] = False


def print_kafka_messages(beamline_acronym_01):
    # print(f"Listening for Kafka messages for {beamline_acronym_01} & {beamline_acronym_02}")
    print(f"Listening for Kafka messages for {beamline_acronym_01}")

    tiled_client = from_profile(beamline_acronym_01)

    # plt.figure()
    # def print_message(name, doc):
    def print_message(consumer, doctype, doc):
        name, message = doc
        print(
            f"\n{datetime.datetime.now().isoformat()} document: {name}\n"
        #     f"\ndocument keys: {list(message.keys())}\n"
        #     f"\ncontents: {pprint.pformat(message)}\n"
            f"\n{message['uid'] = }\n"
        )
        if name == 'start':
            print(
                # f"\n{datetime.datetime.now().isoformat()} documents {name}\n"
                f"\ndocument keys: {list(message.keys())}\n"
                )
                
        elif name == 'event':
            print(
                # f"\n{datetime.datetime.now().isoformat()} documents {name}\n"
                f"\ndocument keys: {list(message.keys())}\n"
                )
                
        elif name == 'stop':
        #     # print('Kafka test good!!')
            print(
                # f"\n{datetime.datetime.now().isoformat()} documents {name}\n"
                f"\ndocument keys: {list(message.keys())}\n"
                f"\ncontents: {pprint.pformat(message['num_events'])}\n"
                f"\n{message['run_start'] = }\n"
                )

            print('\n########### Events printing division ############\n')

    kafka_config = _read_bluesky_kafka_config_file(config_file_path="/etc/bluesky/kafka.yml")

    # this consumer should not be in a group with other consumers
    #   so generate a unique consumer group id for it
    unique_group_id = f"echo-{beamline_acronym_01}-{str(uuid.uuid4())[:8]}"

    kafka_consumer = BasicConsumer(
        topics=[f"{beamline_acronym_01}.bluesky.runengine.documents", 
                # f"{beamline_acronym_02}.bluesky.runengine.documents", 
                ],
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
    # print_kafka_messages(sys.argv[1], sys.argv[2])
    print_kafka_messages(sys.argv[1])
    
