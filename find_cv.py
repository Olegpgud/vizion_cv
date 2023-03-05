import numpy as np
import cv2
import time
 

img_name='имя скриншота'


obj_3_cascade = cv2.CascadeClassifier('haarcascades/cascade_3.xml') # определяем каскад, который будем использовать для поиска бургер-меню с 3 полосками
obj_2_cascade = cv2.CascadeClassifier('haarcascades/cascade_2.xml') # определяем каскад, который будем использовать для поиска бургер-меню с 2 полосками


def find_cv(img_name):
    rgb_img = cv2.imread(img_name) # считываем изображение средствами open_cv

    # уменьшаем размер изображение
    w0, h0 = rgb_img.shape[:-1]
    scale=800/w0
    dsize=(int(h0*scale),int(w0*scale))
    img = cv2.resize(rgb_img, dsize)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # переводим изображение в негатив
    start = time.time()
    obj_ect_3 = obj_3_cascade.detectMultiScale(gray, 1.3, 5) # выполняем поиск объекта по изображению
    obj_ect_2 = obj_2_cascade.detectMultiScale(gray, 1.3, 5) 
    end = time.time() - start
    print(round(end*1000))
    for (x,y,w,h) in obj_ect_3:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2) # выделяем объект на изображении
    for (x,y,w,h) in obj_ect_2:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2) # выделяем объект на изображении
    
    return img 

img=find_cv(img_name)
cv2.imwrite('result.png', img) # сохраняем изображение

