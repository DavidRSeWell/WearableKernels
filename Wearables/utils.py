import io,requests,zipfile

def dowload_zip(data_url,save_name):
  r = requests.get(data_url)
  z = zipfile.ZipFile(io.BytesIO(r.content))
  z.extractall(save_name)
