# -*- coding: utf-8 -*-
import sys

import cv2
from PyQt4 import QtGui


def show_image(image_path='s_pycharm.jpg'):
    """
    # 画像の読み込み
    RGB = 1
    img = cv2.imread(image_path, RGB)

    # 画像の表示
    cv2.imshow('img', img)

    # キーが押させるまで画像を表示したままにする
    # 第一引数：キーイベントを待つ時間　0: 無限, 0以上: 指定ミリ秒待つ
    cv2.waitKey(0)

    # 作成したウィンドウを全て破棄
    cv2.destroyAllWindows()
    """

    app = QtGui.QApplication(sys.argv)
    pixmap = QtGui.QPixmap(image_path)
    screen = QtGui.QLabel()
    screen.setPixmap(pixmap)
    screen.showFullScreen()
    sys.exit(app.exec_())


if __name__ == '__main__':
    show_image()