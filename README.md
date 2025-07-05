This repository contains my work and the final code for my 2nd place result in the "Yale/UNC-CH - Geophysical Waveform Inversion" competition.



The easiest way to run it directly is through this Kaggle notebook: https://www.kaggle.com/code/jeroencottaar/geophysical-waveform-inversion-2nd-place



If you want to make many predictions, you're going to need more compute. I used V100 instances on vast.ai. Specifically:



1. Create a new instance with one or several V100 using the "NVIDIA CUDA" template. Recommended to have at least 250GB to download the competition dataset.



2\. Copy your kaggle.json file into the root directory "/". You can create one here if needed: https://www.kaggle.com/settings



3\. Run the following in a terminal to setup:

cd /

pip install kaggle==1.6.17

pip install kagglehub



kaggle datasets download jeroencottaar/seismic-train-d/ --unzip

mv /kaggle.json /root/.config/kaggle/

chmod 600 /root/.config/kaggle/kaggle.json



sudo apt-get install -y libgl1-mesa-glx

git clone https://github.com/jcottaar/seismic.git /seismic/code

cd /seismic/code



pip install dill

pip install pandas

pip install matplotlib

pip install multiprocess

pip install connected-components-3d

pip install --no-deps monai

pip install scipy

pip install line\_profiler

pip install opencv-python

pip install h5py

pip install timm

pip install scikit-learn

pip install gitpython

pip install seaborn

pip install cupy-cuda12x

pip install portalocker



mkdir /seismic/

mkdir /seismic/data

cd /seismic/data

kaggle competitions download -c waveform-inversion

unzip waveform-inversion.zip

rm waveform-inversion.zip



mkdir /seismic/models

mkdir /seismic/models/brendan

cd /seismic/models/brendan

kaggle datasets download jeroencottaar/brendan-model/ --unzip

