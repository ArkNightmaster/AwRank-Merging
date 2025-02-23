## 文件下载
# import gdown

# url = 'https://drive.google.com/file/d/1uFTzwFc3tmS-D7azjMiJcxSfn71BPqKt/view?usp=sharing'
# output_path = 'graph_ML.pk'
# gdown.download(url, output_path, quiet=False,fuzzy=True)
# https://drive.google.com/drive/folders/10DyLk0jvPB0O-ffU2x1_FGQ2DNJ-_-PY?usp=drive_link
# https://drive.google.com/drive/folders/1HB33xkn9S_FTxUeJ4ax_7jPhZXj2nzsg?usp=drive_link
# 文件夹下载
import gdown
url = "https://drive.google.com/drive/folders/1lVxk6RJW7zG8tebZbHkC3zOLABkMvb4B"

gdown.download_folder(url, quiet=True, use_cookies=False)