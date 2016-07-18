import os
# import cv2                                                                                                                                                                             


def read_image(file_path):
    print(file_path)
    # cv2.read(file_path, 1)  # grayscale                                                                                                                                                
    pass


def traverse_dir(path):
    for file_or_dir in os.listdir(path):
        abs_path = os.path.abspath(os.path.join(path, file_or_dir))
        if os.path.isdir(abs_path):  # dir                                                                                                                                               
            traverse_dir(abs_path)
        else:                        # file                                                                                                                                              
            if abs_path.endswith('pgm'):
                read_image(abs_path)

if __name__ == '__main__':
    traverse_dir("./data")