sudo apt -y install kaggle 
mkdir ./datasets
cd ./datasets

# stanford cars dataset (ref: https://github.com/pytorch/vision/issues/7545#issuecomment-1631441616)
mkdir stanford_cars && cd stanford_cars
# 从Kaggle下载数据集和元数据
kaggle datasets download -d jessicali9530/stanford-cars-dataset
kaggle datasets download -d abdelrahmant11/standford-cars-dataset-meta
# 解压数据集文件
unzip standford-cars-dataset-meta.zip
unzip stanford-cars-dataset.zip
tar -xvzf car_devkit.tgz
# 重组测试集目录结构
mv cars_test a
mv a/cars_test/ cars_test
rm -rf a
# 重组训练集目录结构
mv cars_train a
mv a/cars_train/ cars_train
rm -rf a
# 重命名测试集标注文件
mv 'cars_test_annos_withlabels (1).mat' cars_test_annos_withlabels.mat
# 返回上级目录
cd ..

# ressic45
mkdir resisc45 && cd resisc45
# (manual download) https://onedrive.live.com/?authkey=%21AHHNaHIlzp%5FIXjs&id=5C5E061130630A68%21107&cid=5C5E061130630A68&parId=root&parQt=sharedby&o=OneUp
sudo apt -y install unar
unar NWPU-RESISC45.rar
wget -O resisc45-train.txt "https://huggingface.co/datasets/torchgeo/resisc45/resolve/main/resisc45-train.txt"
wget -O resisc45-val.txt "https://huggingface.co/datasets/torchgeo/resisc45/resolve/main/resisc45-val.txt"
wget -O resisc45-test.txt "https://huggingface.co/datasets/torchgeo/resisc45/resolve/main/resisc45-test.txt"
cd ..

# dtd
mkdir dtd && cd dtd  # 创建并进入 dtd 目录
wget https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz  # 下载数据集
tar -xvzf dtd-r1.0.1.tar.gz  # 解压数据集
mv dtd/images images  # 移动 images 目录到当前目录
mv dtd/imdb/ imdb  # 移动 imdb 目录到当前目录
mv dtd/labels labels  # 移动 labels 目录到当前目录
# 合并训练集和验证集的标签文件
cat labels/train1.txt labels/val1.txt > labels/train.txt
# 复制测试集标签文件
cat labels/test1.txt > labels/test.txt

# euro_sat
mkdir euro_sat && cd euro_sat  # 创建并进入 euro_sat 目录
# 下载数据集（使用 --no-check-certificate 忽略 SSL 证书验证）
wget --no-check-certificate https://madm.dfki.de/files/sentinel/EuroSAT.zip
unzip EuroSAT.zip  # 解压数据集

# sun397
mkdir sun397 && cd sun397  # 创建并进入 sun397 目录
wget http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz  # 下载数据集
unzip Partitions.zip  # 解压分区文件
tar -xvzf SUN397.tar.gz  # 解压主数据集

# Then python preprocess.py on CLI