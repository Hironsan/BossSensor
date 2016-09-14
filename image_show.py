# -*- coding: utf-8 -*-
import sys

from PyQt4 import QtGui


def show_image(image_path='s_pycharm.jpg'):
    app = QtGui.QApplication(sys.argv)
    pixmap = QtGui.QPixmap(image_path)
    screen = QtGui.QLabel()
    screen.setPixmap(pixmap)
    screen.showFullScreen()
    sys.exit(app.exec_())


if __name__ == '__main__':
    show_image()
