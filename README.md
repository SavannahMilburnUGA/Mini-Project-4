# Mini-Project-4
This program analyzes different neural network architectures for classifying ASL fingerspelling shapes accurately. <br>
We use a simple neural network and a deep neural network w/ improvements of cross-entropy cost function, regularization, and better network weight initialization. <br>
We can analyze how certain changes in architecture impact neural network's accuracy <br>

# Understanding Kaggle ASL data structure
* Dataset: https://www.kaggle.com/datasets/datamunge/sign-language-mnist <br>
* Format: label, pixel1, pixel2, ..., pixel784 <br>
* Labels: 0-24 (A-Y. Doesn't have J=9 OR Z=25). <br>
* 24 classes <br>
sign_mnist_train.csv = Training data: <br>
* 27,455 samples w/ 785 columns (1 label + 784 pixels) <br>
sign_mnist_test.csv = Testing data: <br>
* 7172 samples w/ 785 columns (1 label + 784 pixels)