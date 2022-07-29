# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 07:52:50 2022

@author: lenovo
"""
#### importation des librairies

from PIL import Image
import json
import cv2
import numpy as np
import os

#Ouvrir le dossier et parcourir les fichiers
directory = 'C:/Users/lenovo/Downloads/Annotation/ds0/ann'

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)

    if os.path.isfile(f):
        print(f)
     
# Traiter chaque fichier à part

with open("C:/Users/lenovo/Downloads/Annotation/ds0/ann/dent1.jpg.json", "r") as f:
    data = json.load(f)

img = cv2.imread("C:/Users/lenovo/Downloads/Annotation/ds0/img/dent1.jpg")  #lire l'image associée au ficier .json
assert isinstance(img, object)
img_cavity = img.copy()  #créer une copie de l'image originale
edges_list = []

# extraction des contours
for obj in data["objects"]:
    poly_img = np.ones(img.shape, dtype=np.uint8) * 255
    contour_img = np.ones(img.shape, dtype=np.uint8) * 255
    points = (np.array(obj['points']['exterior']))
    print(points)
    # Use fillPoly() function and give input as image,
    # end points,color of polygon
    # Here color of polygon will be green
    cv2.fillPoly(poly_img, pts=[points], color=(0, 0, 0))
    imgray = cv2.cvtColor(poly_img, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(imgray, 100, 200)
    edges_list.append(edged.astype(bool))
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.imshow('Canny Edges After Contouring', edged)
    cv2.drawContours(contour_img, contours, -1, (0, 0, 0), 1)

#différenciation entre mouth_cavity et dents
    if obj['classTitle'] == 'mouth_cavity':
        img_cavity[imgray > 0] = [255, 255, 255]
        cv2.imshow('mouth_cavity', img_cavity)
        cv2.waitKey(0)

edges_list
conc_arr =np.array(edges_list)
print(conc_arr)
sum_arr=np.sum(conc_arr,axis=0)
contour_img = np.ones(img.shape[:2], dtype=np.uint8) * 255
contour_img[sum_arr > 0] = 0
cv2.imshow('contour', contour_img)
image_contours = cv2.cvtColor(contour_img, cv2.COLOR_GRAY2RGB)
#resize de l'image ( pour la division sous forme carrée)
old_image_height, old_image_width, channels = image_contours.shape
padd = abs(old_image_height - old_image_width) // 2
if old_image_height > old_image_width:
    top, bottom, left, right = 0, 0, padd, padd
else:
    top, bottom, left, right = padd, padd, 0, 0
image_contours_ = cv2.copyMakeBorder(image_contours, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                     value=(255, 255, 255))
img_cavity_ = cv2.copyMakeBorder(img_cavity, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(255, 255, 255))
# Image finale
im_final = cv2.hconcat([image_contours_,img_cavity_])
cv2.imshow('im_final', im_final)
# Création des copies d'une image : duplication
# obtention du dataset
for i in range(900,1000):
    cv2.imwrite(f'C:/Users/lenovo/Desktop/Dents/{i}_AB.jpg',im_final )

cv2.waitKey(0)
# Closing all open windows
cv2.destroyAllWindows()