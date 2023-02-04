import cv2
import os
import matplotlib.pyplot as plt
#1.批量读取和保存
root = 'C:/Users/Lenovo/Desktop/pics/The nature/'
file_list = os.listdir(root)
save = "C:/Users/Lenovo/Desktop/pics/edit/"
for name in file_list:
    img_path = root + name
    pht = cv2.imread(img_path, -1)
    out_name = name.split('.')[0]
    save_path = save + out_name + '.png'
    cv2.imwrite(save_path,pht)
#2.缩放（调整分辨率）
img=cv2.imread('C:/Users/Lenovo/Desktop/pics/The nature/p.jpg',-1)
imgd=cv2.resize(img,(30,30))
#3.裁剪
m=img.shape[0]
n=img.shape[1]
img1=img[m//2:m,n//2:n]
#4.直方图
Gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.hist(Gimg)