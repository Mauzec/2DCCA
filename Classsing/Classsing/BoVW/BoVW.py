import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import multiprocessing as mp
import json
from BoVW.sift_cpp.compute import DescriptorSift
from random import choice
from scipy.cluster.vq import kmeans,vq
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from joblib import dump, load

NUM_PROCESS = 8

class BoVW():
    def __init__(self, descriptor = DescriptorSift, code_book = np.ndarray(shape=0),
               number_words = 200, clf = LinearSVC(max_iter=80000), scale: bool=False) -> None:
        self._descriptor = descriptor
        self._image_paths = []
        self._dataset = []
        self._image_classes = []
        self._code_book = code_book
        self._number_words = number_words
        self._stdslr = np.ndarray(shape=0)
        self._clf = clf
        self._class_names = []
        self._scale = scale
        
    def add_train_dataset(self, path: str) -> None:
        
        if not os.path.isdir(path):
            raise NameError("No such directory " + path)
        
        self._class_names = os.listdir(path)
        
        for k, name in enumerate(self._class_names):
            directory = os.path.join(path, name)
            class_path = (os.path.join(directory,f) for f in os.listdir(directory))
            was_len = len(self._image_paths)
            self._image_paths += class_path
            self._image_classes += [k] * (len(self._image_paths) - was_len)
            
        for i in range(len(self._image_paths)):
            self._dataset.append((self._image_paths[i], self._image_classes[i]))
        
    def model_training(self) -> None:
        descriptor_list = self._get_descriptor_list()
        descriptors = descriptor_list[0][1]

        for _, descriptor in descriptor_list[1:]:
            descriptors = np.vstack((descriptors,descriptor))
        descriptors = descriptors.astype(float)
        
        self._code_book, _ = kmeans(descriptors, self._number_words, 1)
        
        image_features = self._get_image_features(descriptor_list)
        
        self._stdslr  = StandardScaler().partial_fit(image_features)
        image_features=self._stdslr.transform(image_features)
        
        self._clf.fit(image_features, np.array(self._image_classes))
        
      
    def testing(self, path_tests: str) -> float:
        self.clear_dataset()
        self.add_train_dataset(path_tests)
          
        descriptor_list_test = self._get_descriptor_list()
            
        test_features = self._get_image_features(descriptor_list_test)
        test_features=self._stdslr.transform(test_features)

        true_classes = []
        count_in_class = [0]*2
        for k in self._image_classes:
            true_classes.append(k)
            count_in_class[k] += 1
            
        predict_classes=[]
        for k in self._clf.predict(test_features):
            predict_classes.append(k)
        
        right_class = [0] * 2
        for k in range(len(predict_classes)):
            if predict_classes[k] == true_classes[k]:
                right_class[true_classes[k]] += 1
                
        accuracy = sum(right_class) / len(true_classes)
        accuracy_c1 = right_class[0] / count_in_class[0] if count_in_class[0] > 0 else 1.0
        accuracy_c2 = right_class[1] / count_in_class[1] if count_in_class[1] > 0 else 1.0
        
        return f"Общая вероятность вывода: {accuracy},\nПравильность определения первого класса: {accuracy_c1},\nправильность определения второго класса: {accuracy_c2}"
        
    def clear_dataset(self) -> None:
        self._image_paths = []
        self._dataset = []
        self._image_classes = []
        self._image_classes_name = []
        
    def classification_image(self, image_path: str) -> tuple[str, int]:
        if not os.path.isfile(image_path):
            return ("no file", -1)

        image = self._image(image_path)
        _, descriptor = self._descriptor.compute(image)
        
        features = np.zeros((1, self._number_words), "float32")

        words, _ = vq(descriptor, self._code_book)
        for w in words:
            features[0][w] += 1

        predicted_class = self._clf.predict(features)[0]

        return (self._class_names[predicted_class], predicted_class)
    
    def _get_descriptor_list(self) -> list:
        descriptor_list = self._parallel_function(self._image_paths, self._get_descriptor)
        delete_index = []
        for k, descriptor in enumerate(descriptor_list):
            if len(descriptor) == 0:
                delete_index.append(k)
            descriptor_list[k] = [self._image_classes[k], descriptor]
        return descriptor_list
    
    def _get_descriptor(self, image_path: str, index_process: int) -> tuple[str, np.ndarray]:
        image = self._image(image_path)
        _, descriptor= self._descriptor.compute(image, index_process=index_process)
        return descriptor
    
    def _get_image_features(self, descriptor_list: list) -> np.ndarray:
        image_features=np.zeros((len(self._image_paths), self._number_words),"float32")
        image_features = self._parallel_function([x[1] for x in descriptor_list], self._get_image_feature)       
        return image_features
    
    def _get_image_feature(self, descriptor: np.ndarray, index_process: int) -> tuple[np.ndarray, int]:
        image_feature = np.zeros(self._number_words,"float32")
        words, _ = vq(descriptor, self._code_book)
        for w in words:
            image_feature[w] += 1 

        return image_feature
    
    @property
    def classes(self) -> dict:
        classes = dict()
        for k, name in enumerate(self._class_names):
            classes[name] = k
            
        return classes
    
    @property
    def dataset(self) -> dict:
        size_classes = dict.fromkeys(self._class_names, 0)
        for k in self._image_classes:
            size_classes[self._class_names[k]] += 1
        
        dataset = {
            "size": len(self._image_paths),
            "size classes": size_classes,
            "words": self._number_words, 
        }
        
        return dataset
    
    @property
    def example(self) -> None:
        image_path = choice(self._image_paths)
        image = self._image(image_path)
        keypoints, _ = self._descriptor.compute(image)
        for keypoint in keypoints:
            x, y = keypoint
            plt.imshow(cv2.circle(image, (int(x), int(y)), 5, (255, 255, 255)))
            
        plt.savefig("example")
        
    def _image(self, image_path: cv2.typing.MatLike) -> cv2.typing.MatLike:
        if self._scale:
            image = cv2.imread(image_path, 0)
            image = cv2.GaussianBlur(image, (5,5), sigmaX=36, sigmaY=36)
            height, width = image.shape
            new_width = min(500, width)
            new_height = int(new_width * (height / width))
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            isWritten = cv2.imwrite(image_path, image)
        return image_path
    
    def save_model(self, name_model = 'modelSVM.jolib', name_classes = "name_classes.json",
                   name_scaler = 'std_scaler.joblib', name_code_book = 'code_book_file_name.npy') -> None:
         
        dump(self._clf, name_model, compress=True)
        dump(self._stdslr, name_scaler, compress=True)
        np.save(name_code_book, self._code_book)
        with open(name_classes, "w") as json_file:
            data = {"names": self._class_names}
            json.dump(data, json_file, ensure_ascii=False)
        
    def download_model(self, name_model = 'modelSVM.jolib', name_classes = "name_classes.json",
                       name_scaler = 'std_scaler.joblib', name_code_book = 'code_book_file_name.npy') -> None:
        
        self._clf = load(name_model)
        self._stdslr = load(name_scaler)
        self._code_book = np.load(name_code_book)
        with open(name_classes, 'r') as json_file: 
            self._class_names = json.load(json_file)["names"]
        
    def _parallel_function(self, data, function) -> list: # в data может быть ndarray или list, в function - функция
        new_data = [None] * len(data)
        input_queue = mp.Queue()
        output_queue = mp.Queue()
        processes = [
            mp.Process(target=self._daemon_function, 
                       args=(input_queue, output_queue, function, i + 1), daemon=True)
            for i in range(NUM_PROCESS)
            ]
        
        for process in processes:
            process.start()
            
        for key, element in enumerate(data):
            input_queue.put((key, element))
            
        k = 0   
        key = None
        element = None
        while k < len(data):
            if not output_queue.empty():
                key, element = output_queue.get()
                new_data[key] = element
                k += 1
            
        for process in processes:
            process.terminate()
            
        return new_data
    
    def _daemon_function(self, input_queue: mp.Queue, output_queue: mp.Queue,
                         function, index_process: int) -> None:
        while True:
            if not input_queue.empty():
                key, input_data = input_queue.get()
                output_data = function(input_data, index_process)
                output_queue.put((key, output_data))