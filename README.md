# Mini-Project-4
This program analyzes different neural network architectures for classifying ASL fingerspelling shapes accurately. <br>
We use a simple neural network, an improved simple neural network w/ improvements of cross-entropy cost function, regularization, and better network weight initialization, and a deep neural network that used the same improved neural network file but is created with multiple hidden layers in the training script. <br>
We also use a CNN; however, due to time constraints, we could not figure out the bug in the learning curves to include all 4 network models so the learning curves only includes the simple, improved, and deep neural network models. <br>
We can analyze how certain changes in neural network architecture impact accuracy via learning curve visualizations along with accuracy and training time visualizations. <br>

# Understanding Kaggle ASL data structure
* Dataset: https://www.kaggle.com/datasets/datamunge/sign-language-mnist <br>
* Format: label, pixel1, pixel2, ..., pixel784 <br>
* Labels: 0-24 (A-Y. Doesn't have J=9 OR Z=25). <br>
* 24 classes <br>
sign_mnist_train.csv = Training data: <br>
* 27,455 samples w/ 785 columns (1 label + 784 pixels) <br>
sign_mnist_test.csv = Testing data: <br>
* 7172 samples w/ 785 columns (1 label + 784 pixels) <br>

AI was used in helping generate visualizations and debugging network examples from the Nielsen textbook.
