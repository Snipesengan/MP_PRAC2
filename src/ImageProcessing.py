import sys
import cv2
import tkinter
import numpy as np
import matplotlib.pyplot as plt

#Excercise 1 - Color Conversion
def excercise_1(img_path):

    img      = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img_hsv  = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    img_luv  = cv2.cvtColor(img,cv2.COLOR_RGB2Luv)
    img_lab  = cv2.cvtColor(img,cv2.COLOR_RGB2Lab)


    plt.figure("Color Conversion")
    plt.subplot(331),plt.imshow(img_gray,cmap='gray'),plt.title('Gray')
    plt.subplot(333),plt.imshow(img_hsv),plt.title('HSV')
    plt.subplot(335),plt.imshow(img),plt.title('RGB')
    plt.subplot(337),plt.imshow(img_luv),plt.title('L*U*V')
    plt.subplot(339),plt.imshow(img_luv),plt.title('L*a*B')

    plt.show()


if __name__ == '__main__':

    if(len(sys.argv) != 3):
        print("Usage -- python {script} <image_path> <excercise #>\n".format(script=sys.argv[0]))
        print(" Excercises")
        print("     1: Color Conversion")
        print("     2: Linear Filtering")
        print("     3: Median Filtering")
        print("     4: Pixel Transform")
        print("     5: Histgoram Equalization")
        print("     6: Morphology")
    else:
        func = [excercise_1]
        func[int(sys.argv[2])-1](sys.argv[1])
