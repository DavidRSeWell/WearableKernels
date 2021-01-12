import os
import pandas as pd

from Wearables.utils import dowload_zip

class WisdmData:

    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00507/wisdm-dataset.zip"
    data_name = "wisdm-dataset"

    @classmethod
    def from_disk(cls,data_path):

        if os.path.isdir(data_path + cls.data_name + "/"):
            return cls(data_path)
        else:
            print("Data does not exist on disk downloading...")
            dowload_zip(cls.URL,data_path)
            return cls(data_path)

    def __init__(self,data_path):

        #self.watch_data = self.load_watch_data(data_path + self.data_name + "/")

        self._data_path = data_path + self.data_name + "/"

    def clean_watch_col(self,row):
        def clean(data):
            if type(data) != str:
                return data
            data = data.strip()
            data = data.replace("\n","")
            data = float(data.replace(";",""))
            return data

        row["x"] = clean(row["x"])
        row["y"] = clean(row["y"])
        row["z"] = clean(row["z"])

        return row

    def load_watch_data(self,data_path):

        watch_data_path = data_path + "raw/watch/"
        accel_data_path = watch_data_path + "accel/"
        gyro_data_path = watch_data_path + "gyro/"

        accel_cols = ["subject_id","activity_code","unix","x","y","z"]
        accel_data = pd.DataFrame([],columns=accel_cols)


        for file in os.listdir(accel_data_path):
            file_df = pd.read_csv(accel_data_path + file, names = accel_cols)
            file_df = file_df.apply(self.clean_watch_col,axis=1)
            accel_data = accel_data.append(file_df)

    def load_subject_watch_data(self,subject_id):
        watch_data_path = self._data_path + "raw/watch/"
        accel_data_path = watch_data_path + "accel/"

        accel_cols = ["subject_id", "activity_code", "unix", "x", "y", "z"]
        file_name = f"data_{subject_id}_accel_watch.txt"
        file_df = pd.read_csv(accel_data_path + file_name, names=accel_cols)
        file_df = file_df.apply(self.clean_watch_col, axis=1)
        return file_df

    def fetch_df_from_file(self,file_path,clean_func= lambda x: x):

        data = []
        lines = open(file_path).readlines()

        for line in lines:
            data.append(clean_func(line))


