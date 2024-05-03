from helpers.image_helper import ImageHelper
from tdpls.tdpls_parallel import TDPLSP
from helpers.correlation_helper import Correlation
import numpy as np
import cv2

D = 10

def get_data(num_test, dataset):
    links_x, links_y = ImageHelper.get_links(num_test=num_test, dataset=dataset)
    X = ImageHelper.get_pictures(links_x)
    Y = ImageHelper.get_pictures(links_y)
    return X, Y, links_x, links_y

def fit_model(X, Y):
    distantion = Correlation.distantion
    algo = TDPLSP(dimension=D, distance_function=distantion, is_max=False)
    algo.fit(X, Y, with_RRPP=True)
    return algo

def predict_and_print(algo:TDPLSP , X_test, train_links_x, test_links_x, test_links_y):
    index_pair = algo.predict(X_test, is_x=True)
    correct_predictions = [index for index in range(len(index_pair)) if index == index_pair[index][0]]
    
    for index in correct_predictions:
        print(train_links_x[index], test_links_x[index])
    
    total_predictions = len(index_pair)
    accuracy = round(len(correct_predictions) / total_predictions, 5)
    
    print(f"Number of correct predictions: {len(correct_predictions)}")
    print(f"Total predictions: {total_predictions}")
    print(f"Accuracy: {accuracy}")
    
    # x_test = X_test[10] - algo.X_c
    # u_pred = algo.W['x'][0].T @ x_test @ algo.W['x'][1]

    # min_distance = np.inf
    # vector = None
    # idx = None
    
    # for i in range(len(algo.U)):
    #     distance = algo.distance_function(u_pred, algo.U[i])
    #     if distance < min_distance:
    #         min_distance = distance
    #         vector = algo.U[i]
    #         idx = i
    # vector = (algo.W['y'][0] @ algo.V[idx] @ algo.W['y'][1].T) + algo.Y_c
    # cv2.imwrite("result.jpg", vector)
    # print(idx)

    
    
    

if __name__ == "__main__":
    X, Y, train_links_x, _ = get_data(num_test=2, dataset="dataset_old")
    algo = fit_model(X, Y)
    X_test, _, test_links_x, test_links_y = get_data(num_test=4, dataset="dataset_old")
    predict_and_print(algo, X_test, train_links_x, test_links_x, test_links_y)


'''
Отчет по лабораторной работе по дисциплине "Алгоритмы и структуры данных" на тему:
    Реализация методов 2D CCA и 2D PLS в приложении к обработке изображений лиц

1. Цель работы
Реализовать программу для распознавания лиц с использованием методов 2D CCA и 2D PLS. 
2. Задачи работы
    Реализовать методы 2D CCA и 2D PLS.
    Применить редукцию размерности пространства признаков к методам(РРПП) 2D CCA и 2D PLS.
    Применить методы 2D CCA и 2D PLS к задаче распознавания лиц.
    Реализовать программу для распознавания лиц.
    Провести эксперименты с использованием методов 2D CCA и 2D PLS.
    
3. Теоритическая часть
3.1. Описание задачи
    Задача распознавания лиц заключается в поиске сходства между двумя изображениями лиц. 
    В данной работе рассматриваются два набора изображений лиц: изображения в термальном диапазоне и изображения в видимом диапазоне.
    Задача состоит в том, чтобы определить по изображению в термальном диапазоне изображение в видимом диапазоне, и наоборот.
'''