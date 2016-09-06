# -*- coding: utf-8 -*-
import tensorflow as tf
from PIL import Image
import os
import cv2


def read_image(file_path):
    image = cv2.imread(file_path, 0)  # grayscale
    print(image)
    print(image.shape)
    return image

images = []
labels = []
def traverse_dir(path, sess):
    for file_or_dir in os.listdir(path):
        if len(images) > 100:
            return images, labels
        abs_path = os.path.abspath(os.path.join(path, file_or_dir))
        print(abs_path)
        if os.path.isdir(abs_path):  # dir
            traverse_dir(abs_path, sess)
        else:                        # file
            if file_or_dir.endswith('.jpg'):
                image = read_image_(abs_path, sess)
                images.append(image)
                labels.append(path)
    return images, labels



#csv_name = 'path/to/filelist.csv'
#fname_queue = tf.train.string_input_producer([csv_name])
#reader = tf.TextLineReader()
#key, val = reader.read(fname_queue)
#fname, label = tf.decode_csv(val, [["aa"], [1]])
IMAGE_SIZE = 96
def read_image_(file_path, sess):
    # 既存ファイルを readモードで読み込み
    #img = Image.open(file_path, 'r')

    # resizeではなくthumbnailを利用して縮小。画像が小さい場合は大きくなる
    #img.thumbnail((IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)

    # リサイズ後の画像を保存
    #img.save('tmp.jpg', 'JPEG', quality=100, optimize=True)

    #jpeg_r = tf.read_file('tmp.jpg')
    jpeg_r = tf.read_file(file_path)
    image = tf.image.decode_jpeg(jpeg_r, channels=3)
    image.set_shape(sess.run(image).shape)
    h, w, _ = image.get_shape()
    longest_edge = int(max(h, w))
    image = tf.image.resize_image_with_crop_or_pad(image, longest_edge, longest_edge)
    image = tf.image.resize_image_with_crop_or_pad(image, IMAGE_SIZE, IMAGE_SIZE)
    return image

#traverse_dir('./data')

if __name__ == '__main__':
    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)
    tf.train.start_queue_runners(sess)
    #images, labels = traverse_dir('data', sess)

"""
image = read_image_('./data/boss/27.jpg', sess)
x = sess.run(image)
print(x)
print(x.shape)
import matplotlib.pyplot as plt
plt.imsave("test.jpg", x)
"""


"""
from PIL import Image

# 既存ファイルを readモードで読み込み
img = Image.open('./data/boss/2.jpg', 'r')

# resizeではなくthumbnailを利用して縮小。画像が小さい場合は大きくなる
img.thumbnail((100, 100), Image.ANTIALIAS)

# リサイズ後の画像を保存
img.save('thumbnail_img.jpg', 'JPEG', quality=100, optimize=True)


# 既存画像を読み込み
a_jpg = Image.open('./data/boss/64_c.jpg', 'r')

# マージに利用する下地画像を作成する
canvas = Image.new('RGB', (180, 180), (0, 0, 0))

# pasteで、座標（0, 0）と（0, 100）に既存画像を乗せる。
canvas.paste(a_jpg, (0, 0))

# 保存
canvas.save('c.jpg', 'JPEG', quality=100, optimize=True)
"""