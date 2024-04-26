from helpers.image_helper import ImageHelper
from tdpls.tdpls_parallel import TDPLSP
from helpers.correlation_helper import Correlation

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

def predict_and_print(algo, X_test, test_links_x, test_links_y):
    index_pair = algo.predict(X_test, is_x=True)
    correct_predictions = [index for index in range(len(index_pair)) if index == index_pair[index][0]]
    
    for index in correct_predictions:
        print(test_links_x[index], test_links_y[index])
    
    total_predictions = len(index_pair)
    accuracy = round(len(correct_predictions) / total_predictions, 5)
    
    print(f"Number of correct predictions: {len(correct_predictions)}")
    print(f"Total predictions: {total_predictions}")
    print(f"Accuracy: {accuracy}")

if __name__ == "__main__":
    X, Y, _, _ = get_data(num_test=5, dataset="dataset_old")
    algo = fit_model(X, Y)
    X_test, _, test_links_x, test_links_y = get_data(num_test=3, dataset="dataset_old")
    predict_and_print(algo, X_test, test_links_x, test_links_y)
