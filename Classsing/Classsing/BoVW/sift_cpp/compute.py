import os
import json
from platform import platform as pf
is_win = pf().startswith('Win')
entry_path = f".\BoVW\\sift_cpp\\main.exe" if is_win else f"{os.getcwd()}/BoVW/sift_cpp/main"

class DescriptorSift:
    @staticmethod
    def compute(image_path: str, index_process = -1, drawkps: int = 0):
        ''' Search keypoints and descriptors. 
            Usage example:
                from sift_cpp.compute import DescriptorSift

                kps, des = DescriptorSift.compute('book.png', drawkps=0)
                #kps, des = DescriptorSift.compute('book.png', drawkps=1) for create result.jpg with keypoints marked
        '''
        assert drawkps == 0 or drawkps == 1
        os.system(f'{entry_path} {image_path} -drawkps={drawkps} {index_process}')

        data = None
        with open(f"kps{index_process}.json", 'r') as file: data = json.load(file)
        os.remove(f"kps{index_process}.json")
        kps = []
        des = []
        for item in data['kpsdes']:
            kps.append((item["x"], item["y"]))
            des.append(item['des'])
        return kps, des

if __name__ == "__main__":
    DescriptorSift.compute("book_in_scene.jpg")

        

