from BoVW.BoVW import BoVW
from BoVW.dataset.get_pictures import Dataset_operations
from BoVW.sift_cpp.compute import DescriptorSift

def main(percentage: int):
    # Dataset_operations.clear()
    # Dataset_operations.get_mona_original()
    # Dataset_operations.get_work_train_dataset(percentage_train=percentage)
    # Dataset_operations.get_images(start=50, end=70, for_using="test", similar=True)
    
    bovw = BoVW(scale=True, descriptor=DescriptorSift, number_words=500)
    
    # print("start add dataset")
    # bovw.add_train_dataset("BoVW/dataset/train")
    # print("end add dataset")
    
    # print("start training model")
    # bovw.model_training()
    # print("end training model")
    
    # print("save model")
    # bovw.save_model()
    
    # print("start testing")
    # print(bovw.testing("BoVW/dataset/test"))
    # print("end testing")
    
    bovw.download_model()
    Dataset_operations.get_mona_original()
    Dataset_operations.get_mona_younger()
    
    print("Оригинальная: ", bovw.classification_image("BoVW/dataset/train/artist/mona_original.png"))
    print("Айзелуорсткая: ", bovw.classification_image("BoVW/dataset/test/artist/mona_younger_1.jpg"))
    print("Эрмитажная: ", bovw.classification_image("BoVW/dataset/test/artist/mona_younger_2.jpg"))
    
    Dataset_operations.clear()
    
    print("end program")
    
if __name__ == "__main__":
    main(30)