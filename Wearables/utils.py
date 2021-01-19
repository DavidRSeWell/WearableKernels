import io
import os
import requests
import yaml
import zipfile


def dowload_zip(data_url,save_name):
  r = requests.get(data_url)
  z = zipfile.ZipFile(io.BytesIO(r.content))
  z.extractall(save_name)

def load_config(name: str) -> dict:

  if not os.path.isfile(name):
    try:
      os.chdir("../")
      path = os.getcwd()
      name = path + "/etc/" + name
      assert os.path.isfile(name)
    except Exception as e:
      print(f"Could not find the config file {name}")
      raise

  with open(name) as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
    return data