import json
import pandas as pd

from sklearn.model_selection import train_test_split

from Wearables import utils
from Wearables.features import _downsample_mean
from Wearables.kernel_methods import W2VKernel
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

    watch_accel_data, watch_gyro_data, phone_accel_data, phone_gyro_data = WisData.load_data()

    watch_accel_data['datetime'] = pd.to_datetime(watch_accel_data['unix'])
    watch_accel_data = watch_accel_data.drop('unix', axis=1)
    watch_accel_data = watch_accel_data.set_index(['datetime'])

    # Perform Transformation
    trans_data = _downsample_mean(watch_accel_data)

    X, y = trans_data[['x', 'y', 'z','subject','activity']], trans_data[['subject','activity']]

    model = W2VKernel(n_clusters=30)

    model.run(X,y)
    print("Done running")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    import os
    curr_path = os.getcwd()
    data_path = curr_path + "/config.yaml"
    run(data_path)

    print("Done running main")


