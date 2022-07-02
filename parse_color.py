import PIL
from PIL import Image
import cv2
import sklearn
from sklearn.cluster import KMeans
import numpy as np

def get_main_color_list_img(img_path):
    """
    対象の画像のメインカラーを算出し、色を横並びにしたPILの画像を取得する。

    Parameters
    ----------
    img_path : str
        対象の画像のパス。

    Returns
    -------
    tiled_color_img : Image
        色を横並びにしたPILの画像。
    """
    cv2_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    print(cv2_img.shape)
    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    # testImg = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2HSV)
    cv2_img = cv2_img.reshape(
        (cv2_img.shape[0] * cv2_img.shape[1], 3))

    cluster = KMeans(n_clusters=1)
    cluster.fit(X=cv2_img)
    cluster_centers_arr = cluster.cluster_centers_.astype(
        int, copy=False)

    # removeBG = cv2.inRange(testImg, np.array([207, 55, 50]), np.array([227, 75, 70]))
    # cv2.namedWindow("mask", flags = cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
    # cv2.imshow('mask', removeBG)
    # cv2.waitKey(0)
    # cv2.imshow('2.png', removeBG)
    # removeBG.save("2.png")


    IMG_SIZE = 64
    MARGIN = 15
    width = IMG_SIZE * 5 + MARGIN * 2
    height = IMG_SIZE + MARGIN * 2

    tiled_color_img = Image.new(
        mode='RGB', size=(width, height), color='#333333')
    for i, rgb_arr in enumerate(cluster_centers_arr):
        color_hex_str = '#%02x%02x%02x' % tuple(rgb_arr)
        print(color_hex_str)
        color_img = Image.new(
            mode='RGB', size=(IMG_SIZE, IMG_SIZE),
            color=color_hex_str)
        tiled_color_img.paste(
            im=color_img,
            box=(MARGIN + IMG_SIZE * i, MARGIN))
    return tiled_color_img

get_main_color_list_img(img_path='./results/favicon_out.png').save("./1.png")