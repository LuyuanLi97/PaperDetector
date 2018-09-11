from sklearn import datasets, svm, metrics
from sklearn.datasets import fetch_mldata
from sklearn.externals import joblib

MODEL_PATH = 'mnist_svm_model_full.pkl'

mnist = fetch_mldata('MNIST original', data_home='./scikit_learn_data')
X_data = mnist.data / 255.0
Y = mnist.target
# print('svm')
classifier = svm.SVC(C=5, gamma=0.05)
classifier.fit(X_data, Y)
joblib.dump(classifier, MODEL_PATH, compress=3)