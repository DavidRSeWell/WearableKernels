import os
import pandas as pd
import time

from Wearables.utils import dowload_zip
from typing import List,Dict

class WisdmData:

    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00507/wisdm-dataset.zip"
    data_name = "wisdm-dataset"

    @classmethod
    def from_config(cls,config: Dict):

        data_path = config["DATA"]["data_path"]

        if os.path.isdir(data_path + cls.data_name + "/"):
            return cls(**config["DATA"])
        else:
            print("Data does not exist on disk downloading...")
            dowload_zip(cls.URL,data_path)
            return cls(**config["DATA"])

    def __init__(self,data_path: str,load_watch: dict = True,load_phone: dict = False,
                 load_all: bool = True, load_subjects: List = None):

        #self.watch_data = self.load_watch_data(data_path + self.data_name + "/")

        self._data_path = data_path + self.data_name + "/"
        self._load_all = load_all
        self._load_phone = load_phone
        self._load_subjects = load_subjects
        self._load_watch = load_watch

        self.subject_ids = self.load_subject_ids()

    def clean_device_col(self,row):
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

    def load_data(self) -> tuple:
        """
        Load data based on the class variables
        :return:
        """
        subject_ids = self.subject_ids
        if not self._load_all:
            subject_ids = list(self._load_subjects)

        watch_accel_data,watch_gyro_data = self.load_device_data(device_type="watch",subjects=subject_ids)

        phone_accel_data,phone_gyro_data = self.load_device_data(device_type="phone",subjects=subject_ids)

        return (watch_accel_data,watch_gyro_data,phone_accel_data,phone_gyro_data)

    def load_device_data(self,device_type:str,subjects: List) -> tuple:

        print(f"Loading data for {device_type}")
        load_accel, load_gyro = tuple(self._load_watch.values())
        if device_type == "phone":
            load_accel, load_gyro = tuple(self._load_phone.values())

        accel_cols = ["subject_id","activity_code","unix","x","y","z"]
        accel_data = pd.DataFrame([],columns=accel_cols)

        s_time = time.time()

        if load_accel:
            for subject in subjects:
                assert int(subject)
                file_df = self.load_subject_data(int(subject),device_type,"accel")
                print(f"Data has shape {file_df.shape}")
                accel_data = accel_data.append(file_df)

        print(f"Done loading accel data shape {accel_data.shape}")
        e_time = time.time()
        r_time = (e_time - s_time) / 60.0
        print(f"runtime = {r_time}")

        s_time = time.time()
        gyro_data = pd.DataFrame([],columns=accel_cols)

        if load_gyro:
            for subject in subjects:
                assert int(subject)
                file_df = self.load_subject_data(int(subject), device_type, "gyro")
                print(f"Data has shape {file_df.shape}")
                gyro_data = gyro_data.append(file_df)

        print(f"Done loading gyro data shape {gyro_data.shape}")
        e_time = time.time()
        r_time = (e_time - s_time) / 60.0
        print(fr"runtime = {r_time}")

        return accel_data,gyro_data

    def load_subject_data(self,subject_id: int,device_type: str,data_type: str):

        device_folder_path = self._data_path + f"raw/{device_type}/"
        data_path = device_folder_path + f"{data_type}/"

        cols = ["subject_id", "activity_code", "unix", "x", "y", "z"]
        file_name = f"data_{subject_id}_{data_type}_{device_type}.txt"
        file_df = pd.read_csv(data_path + file_name, names=cols)
        file_df = file_df.apply(self.clean_device_col, axis=1)
        return file_df

    def load_subject_ids(self):

        device_data_path = self._data_path + f"raw/watch/"
        accel_data_path = device_data_path + "accel/"
        subject_ids = []

        for file in os.listdir(accel_data_path):
            sub_id = file.split("_")[1]
            try:
                sub_id = int(sub_id)
            except:
                print(f"Could not process id for file {file}")
                continue
            subject_ids.append(sub_id)

        return subject_ids


    def fetch_df_from_file(self,file_path,clean_func= lambda x: x):

        data = []
        lines = open(file_path).readlines()

        for line in lines:
            data.append(clean_func(line))


