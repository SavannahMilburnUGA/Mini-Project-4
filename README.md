# Mini-Project-4
ASL sign classification <br>
https://www.kaggle.com/datasets/datamunge/sign-language-mnist - Only fingerspelling signs and 0-9

# Understanding Kaggle data structure
* Format: label, pixel1, pixel2, ..., pixel784
* Labels: 0-24 (A-Y. Doesn't have J=9 OR Z=25). 
* 24 classes <br>
sign_mnist_train.csv = Training data: <br>
* 27,455 samples w/ 785 columns (1 label + 784 pixels) <br>
sign_mnist_test.csv = Testing data: <br>
* 7172 samples w/ 785 columns (1 label + 784 pixels)
