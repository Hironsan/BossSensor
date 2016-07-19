import os
import cv2


def read_image(file_path):
    image = cv2.imread(file_path, 0)  # grayscale
    print(image)
    print(image.shape)
    return image

images = []
labels = []
def traverse_dir(path):
    for file_or_dir in os.listdir(path):
        abs_path = os.path.abspath(os.path.join(path, file_or_dir))
        if os.path.isdir(abs_path):  # dir
            traverse_dir(abs_path)
        else:                        # file
            if file_or_dir.endswith('.pgm'):
                image = read_image(abs_path)
                images.append(image)
                labels.append(path)
    return images, labels

if __name__ == '__main__':
    images, labels = traverse_dir("./data")
    print(len(images))
    print(len(labels))
    print(labels)
    # print(images.shape)
    # print(labels.shape)