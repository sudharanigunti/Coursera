import tensorflow as tf

# object recognition
#Take input Image
# Extract the features
# Parts are identified
# join parts and detect/label the full body in image.

# horizontal and vertical lines in the image, for identifying the building, windows.
# faces, animals, cars

# hand written digits - Digit recognition system using CNN- mnist dataset
# shallow neural network- has limitation of feature extraction
# CNN = automatically find features and uses it for finding output.

# PHASES - preprocessing, training the model, inference phase evaluating.
# Hidden layers = convlutional layer + pooling layer + fully connected layer

# Architecture of the CNN
# Convolution layer = detect the features from the image
# ex: edges of the image , apply filter by roating the image thus we get top view of the image in 2d lines boader.
# its a mathematical func 
# USe edge detector filter and SLide the image= apply dot product
# Features = edges, curves and so on
# Feature Maps

# Initialze the kernal with random values and it optimzes in training.
# Each kernel will recognize a particular pattern in the image.
# more the kernels more the different patterns in the images.
#  ReLu= rectified linear unit... it is a non-linear function

# Downsample the image, max-pooling = reduces the number of pararmeters in the data.
# Stride = 2, then window will move 2 pixels everytime.

# fully - connected
# Softmax is an activation func, used in output layer to find the class.
# multi class probability layer.


# Reduces the height and width -> increases the depth of the image.
# Layer 1 - details of each Part
# Layer 2 - shows the node, ears parts
# Layer 3 - shows the face like patterns


# conv -> ReLu -> MaxPool -> Conv -> ReLu -> MaxPool -> FullyConnected -> ReLu -> FullyConnected -> ReLu -> Output
