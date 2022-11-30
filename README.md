# learning_based_fall_detection

Github Repo for development on ECE 598JK final project - learning based fall detection.

## Setup
```
git clone https://github.com/Lionel-Lee/learning_based_fall_detection.git
cd ~/learning_based_fall_detection
virtualenv -p /usr/bin/python3 myenv
source myenv/bin/activate
pip install -r requirements.txt
```

## Data

The data are collected from the MINI robot IMU sensor, and stored in the ./data directory.

During training and evaluation, the data is loaded with the traj_data_loader dataset wrapper.

The training observed trajectory data have shape (N,T,motion_dim), where N is the number of data samples, T is the sequence length of the observation, and motion_dim is the dimension of IMU data.

The training labels have shape (N,1), i.e., one label per data sample. label = 1 indicates a falling process and 0 otherwise.


## Model Backbone

The model backbone is written in ./model/fall_detection_lstm.py. The learning model is based on one-directional LSTM. The embedding size, lstm hidden size and number of hidden layers are specified by the arguments parsing function in ./utils/args.py

Besides the lstm backbone, there is also a input embedding layer and an output head. The input embedding layer augment the input data dimension to the hidden size of the lstm units, while the output head reduces the size to 1 for sigmoid activation and binary crossentropy classification.

The prediction is based on the hidden states output of the lstm units, not the cell states.


## Training and Evaluation

To train/evaluate the model, the mode should be set to 'train' and 'eval' respectively. 

```
python main.py --mode train --data_file_path data/imu_train.txt
python main.py --mode eval --data_file_path data/imu_eval.txt
```

Both the training and evaluation pipeline are written in ./main.py. The trianing number of epochs, leraning rate, leraning rate scheduler, model saving path, etc. are all specified in ./utils/args.py. Trained models are stored as .pt files in the ./trained_model/ directory.

## Deployment
During the locomotion or any other actions of the robot, an independent thread is tracking the IMU data, extracting 10 dimensions (angular velocity, acceleration, orientation, etc.), applying a low-pass Kalman Filter, and writing the processed data into a local data file.

The remote device keeps polling the data file at a certain frequency (50 Hz), parsing the data file and feed the data into the neural network for prediction.  

To activate deployment on the remote device side, run:
```
python main.py --mode deploy
```

