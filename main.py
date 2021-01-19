import json
import pandas as pd

from Wearables import utils
from Wearables.kernel_methods import TransferKernel
from Wearables.wisdm import WisdmData


def run(config):

    if type(config) == dict:
        print("Assuming passed in config is a correctly formatted dictionary")
    else:
        print("Assuming path to config has been passed")
        config = utils.load_config(config)

    print("CONFIG")
    print("----------------------------------------------------------")
    print(json.dumps(config, indent=4, sort_keys=True))

    WisData = WisdmData.from_config(config)

    data = WisData.load_data()

    print("Done running")



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    import os
    curr_path = os.getcwd()
    data_path = curr_path + "/config.yaml"
    run(data_path)

    print("Done running main")


