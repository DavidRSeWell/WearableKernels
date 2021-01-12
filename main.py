import pandas as pd

from Wearables.kernel_methods import TransferKernel
from Wearables.wisdm import WisdmData

def run(data_path):

    WisData = WisdmData.from_disk(data_path)

    watch_data = WisData.load_subject_watch_data(1600)

    transfer_kernel = TransferKernel(watch_data)
    print("Done running")



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data_path = ""
    run(data_path)

    print("Done running main")


