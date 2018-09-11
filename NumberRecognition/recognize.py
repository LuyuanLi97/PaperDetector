import cv2
import numpy as np
import os
import os.path
from sklearn.externals import joblib
import csv

READ_FOLDER = 'input'
DILATION_FOLDER = 'dilation'
LINE_CUT_FOLDER = 'line_cut'
LINE_FOLDER = 'line'
DIGIT_FOLDER = 'digit'
MODEL_PATH = 'mnist_svm_model_full.pkl'
RESULT_FOLDER = 'result'

MNIST_SIZE = 28

if not os.path.exists(DILATION_FOLDER):
    os.makedirs(DILATION_FOLDER)
if not os.path.exists(LINE_CUT_FOLDER):
    os.makedirs(LINE_CUT_FOLDER)
if not os.path.exists(LINE_FOLDER):
    os.makedirs(LINE_FOLDER)
if not os.path.exists(DIGIT_FOLDER):
    os.makedirs(DIGIT_FOLDER)
if not os.path.exists(RESULT_FOLDER):
    os.makedirs(RESULT_FOLDER)

kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
kernel_cross_h = cv2.getStructuringElement(cv2.MORPH_CROSS, (1, 8))
kernel_cross_w = cv2.getStructuringElement(cv2.MORPH_CROSS, (8, 1))
kernel_connect = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)

'''
def is_written_vertically(img):
    h, w = img.shape
    h_bin = np.zeros(h, np.uint16)
    w_bin = np.zeros(w, np.uint16)
    y, x = np.where(img == 255)
    for i in y:
        h_bin[i] += 1
    for i in x:
        w_bin[i] += 1

    n_h_zero_area = 0
    for i in range(h - 1):
        if h_bin[i] == 0 and h_bin[i + 1] != 0:
            n_h_zero_area += 1
    n_w_zero_area = 0
    for i in range(w - 1):
        if w_bin[i] == 0 and w_bin[i + 1] != 0:
            n_w_zero_area += 1

    if n_h_zero_area > n_w_zero_area:
        return False
    return True
'''

# filename: picture which is binary
def find_digits_and_predict(filename):
    # for filename in os.listdir(folder):
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        fn = filename.split('\\')[1]
        # split file name and extension name
        fn = os.path.splitext(fn)[0]
        if img is not None:
            height, width = img.shape
            line_name = os.path.join(LINE_CUT_FOLDER, fn + '_line.bmp')
            dilation_name = os.path.join(DILATION_FOLDER, fn + '_dil.bmp')
            # is_vertical = is_written_vertically(img)
            # print(is_vertical)

            dilated = cv2.dilate(img, kernel_ellipse, iterations=1)
            # if is_vertical:
            #     dilated = cv2.dilate(dilated, kernel_cross_h, iterations=10)
            # else:
            dilated = cv2.dilate(dilated, kernel_cross_w, iterations=10)

            _, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            predict_result_whole_paper = list()
            # index
            i = 0
            for contour in contours:
                [x, y, w, h] = cv2.boundingRect(contour)
                # if is_vertical and (w < 30 or w > 100 or h < 70 or h > 520):
                #     continue
                if h < 10 or w < 30:
                    continue

                i += 1
                digits_line = img[y:y + h, x:x + w]
                save = np.rot90(digits_line)
                cv2.imwrite(os.path.join(LINE_FOLDER, str(i) + 'str.bmp'), save)
                digits_arr = split_digits_from_line(digits_line, fn + '_s' + str(i))
                # predict this digit array
                predicted = predict_mnist_svm(digits_arr)
                predict_result_whole_paper.append(predicted)

                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)

            # print predict_result_whole_paper in a reversed way
            print(fn + ': ')
            for line in predict_result_whole_paper[::-1]:
                print line
                # write result to csv
                filename = fn + ".csv"
                with open(os.path.join(RESULT_FOLDER, filename), "a") as f:
                    np.savetxt(f, line, fmt="%d", delimiter='', newline='')
                    f.write('\n')
            # print('\n')

            cv2.imwrite(line_name, img)
            cv2.imwrite(dilation_name, dilated)


def split_digits_from_line(s, prefix_name):
    # print(is_vertical)

    # if is_vertical:
    #    s = np.rot90(s, 2)
    # else:
    s = np.rot90(s)

    # dilate in case the digits are not continuous
    s_copy = cv2.dilate(s, kernel_connect, iterations=1)

    s_copy2 = s_copy.copy()
    _, contours, hierarchy = cv2.findContours(s_copy2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # index
    idx = 0
    digits_arr = np.array([])
    for contour in contours:
        idx = idx + 1
        [x, y, w, h] = cv2.boundingRect(contour)
        digit = s_copy[y:y + h, x:x + w]

        padding_len = (h - w) / 2
        if padding_len > 0:
            digit = cv2.copyMakeBorder(digit, 0, 0, padding_len, padding_len, cv2.BORDER_CONSTANT, value=0)
        elif padding_len < 0:
            digit = cv2.copyMakeBorder(digit, -padding_len, -padding_len, 0, 0, cv2.BORDER_CONSTANT, value=0)

        # print(digit.shape)
        extra_pad = digit.shape[0] / 5
        digit = cv2.copyMakeBorder(digit, extra_pad, extra_pad, extra_pad, extra_pad, cv2.BORDER_CONSTANT, value=0)

        digit = np.rot90(digit, 3)
        # print(digit.shape)
        digit = cv2.resize(digit, (MNIST_SIZE, MNIST_SIZE))
        digit_name = os.path.join(DIGIT_FOLDER, prefix_name + '_n' + str(idx) + '.bmp')
        cv2.imwrite(digit_name, digit)

        digit = np.concatenate([(digit[i]) for i in range(MNIST_SIZE)])
        digits_arr = np.append(digits_arr, digit)

        digits_arr = digits_arr.reshape((digits_arr.shape[0] / (MNIST_SIZE * MNIST_SIZE), -1))
    # print(digits_arr.shape)
    # print(digits_arr)
    return digits_arr


def train_mnist_svm():
    if os.path.isfile(MODEL_PATH):
        classifier = joblib.load(MODEL_PATH)
    return classifier


def predict_mnist_svm(digits_arr):
    classifier = train_mnist_svm()
    digits_arr = digits_arr / 255.0
    predicted = classifier.predict(digits_arr)
    # print(predicted)
    return predicted


for filename in os.listdir(READ_FOLDER):
    find_digits_and_predict(os.path.join(READ_FOLDER, filename))