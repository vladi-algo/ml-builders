# ml-builders
cnn_training_model.py - code for training the model for gesture prediction

predictor.py - code for gesture perdiction 

print_images.py - code for creaiting images (new gestures) for training 


## Set up
Create python virtual environment and install following dependencies:

yum install mesa-libGL 

pip install tensorflow

pip install numpy

pip install keras

pip install opencv-python

pip install matplotlib

pip install pandas

pip install scikit-learn

Clone the project.

Prepare training data by using print_images.py

Run training: python3 cnn_training_model.py

After model is ready predict with predictor.py


## generate gestures
In order to generate gestures run create_gesture_data.py 
-g parameter is train or test
-i parameter is a gesture id. 0,1,2,3,4,.. 10
For example : create_gesture_data.py -i 1 -g train - creates images for gesture 1 and copy them to the train folder

## training 
In order to train model run DataFlair_trainCNN.py

## prediction 
In order to predict run model_for_gesture.py