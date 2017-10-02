# Lung Cancer Classification on CT Scan Images Using a 3D Convolutional Neural Network

![header](https://www.kaggle.io/svf/1028993/84c1f676416439fdd579990e4105d8c7/__results___files/__results___7_1.png)

The goal of this project was to detect the presence of lung cancer nodules from a large dataset of CT scan images. The dataset used contained CT scan images for over 1500 patients and contained respective ground truth values. From this dataset a 3D convolutional neural network was trained and applied to the data, resulting in an average accuracy of 0.589 which is comparable with other modern classification approaches on the same dataset.

## Datasets

Datasets, and project problem, are taken from the [Kaggle Data Science Bowl 2017](https://www.kaggle.com/c/data-science-bowl-2017/data).

During development we used the sample_images/ dataset since it only contains 20 patients, for our actual results however we used the stage1/ dataset which contains close to 1500 usable patients.

## How to Run

First you need to run `preprocess.py` in order to produce `processed_data.npy`. Once you have built `processed_data.npy` you can run `cnn.py`. If you have the option to use tensorflow on your GPU everything will run MUCH faster. It takes ~320 seconds to run `cnn.py` with 10 epochs on a GTX 1070 with 8.00GiB of ram and a memory clock rate of 1.7715 GHz.

## Results

Best output obtained from running `conv_nn.py` with 10 epochs on a GTX 1070.

`Finishing accuracy:`

`Accuracy: 0.67`

`fitment percent: 1.0`

`Total running time: 321.9375514984131 seconds.`

After running for 10 different trials our average accuracy came out to `0.589`, this is further discussed in our report.

## Dependencies

In order to run our code on GPU to obtain similar results you will need the following:
- The `stage1/` dataset from [Kaggle](https://www.kaggle.com/c/data-science-bowl-2017/data)
- [Python 3.5.x 64 bit](https://www.python.org/downloads/release/python-352/) - it is critical you install the 64 bit version not the 32 bit version
- [TensorFlow 1.0 GPU Installation](https://www.tensorflow.org/install/install_windows) - `pip3 install --upgrade tensorflow-gpu`
- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org/)
- dicom - `pip install --upgrade dicom`
- cv2 - `pip install --upgrade opencv`

## References

The following are the major tutorials/blog posts that helped us figure out how to create a CNN using TensorFlow:
- [Official TensorFlow CNN Tutorial](https://www.tensorflow.org/tutorials/deep_cnn)
- [pythonprogramming blog](https://pythonprogramming.net/)
- [pythonprogramming neural networks and machine learning tutorial](https://pythonprogramming.net/neural-networks-machine-learning-tutorial/)
- [pythonprogramming TensorFlow and intro to machine learning tutorial](https://pythonprogramming.net/tensorflow-introduction-machine-learning-tutorial/)
- [pythonprogramming cnn machine learning tutorial](https://pythonprogramming.net/convolutional-neural-network-cnn-machine-learning-tutorial/)
- [pythonprogramming cnn TensorFlow tutorial](https://pythonprogramming.net/cnn-tensorflow-convolutional-nerual-network-machine-learning-tutorial/)
