import sys
import cv2
import tkinter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons


#Excercise 1 - Color Conversion
def excercise_1(img_path):

    img      = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img_hsv  = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    img_luv  = cv2.cvtColor(img,cv2.COLOR_RGB2Luv)
    img_lab  = cv2.cvtColor(img,cv2.COLOR_RGB2Lab)


    #Note that this is not the point, the image should be the same
    #regardless of the color space (of course for gray), as there
    #shouldn't be any information lost.

    plt.figure("Color Conversion")
    plt.subplot(331),plt.imshow(img_gray,cmap='gray'),plt.title('Gray')
    plt.subplot(333),plt.imshow(img_hsv,cmap='hsv'),plt.title('HSV')
    plt.subplot(335),plt.imshow(img),plt.title('RGB')
    plt.subplot(337),plt.imshow(img_luv),plt.title('L*U*V')
    plt.subplot(339),plt.imshow(img_luv),plt.title('L*a*B')

    plt.show()

#Excercise 2 - Linear Filtering
def excercise_2(img_path):
    sigma = 0
    kernel = 3

    #lets define a bunch of kernels
    prewit = np.array([[-1,0,1],[-1,0,1],[-1,0,1]]).T
    sobel  = prewit.T
    laplacian = np.array([[0,1,0],[1,-4,1],[0,1,0]])

    img        = cv2.imread(img_path,0) #read as a grey scale image
    prewit_img = cv2.filter2D(img,-1,prewit)
    sobel_img  = cv2.filter2D(img,-1,sobel)
    lap_img = cv2.filter2D(img,-1,laplacian)
    gauss_img = cv2.GaussianBlur(img,(3,3),sigma)

    plt.figure("Prewit Filter")
    plt.imshow(np.hstack((img,prewit_img)),cmap='gray')
    plt.figure("Sobel Filter")
    plt.imshow(np.hstack((img,sobel_img)),cmap='gray')
    plt.figure("Laplacian Filter")
    plt.imshow(np.hstack((img,lap_img)),cmap='gray')


    #I want to be sweaty for this one, and add a slider to change
    #kernel size, sigmaX and sigmaY

    plt.figure("Gaussian Filter")
    fig = plt.imshow(np.hstack((img,gauss_img)),cmap='gray')

    #Create axis for sigmaX
    axcolor  = 'lightgoldenrodyellow'
    axsigma = plt.axes([0.25,0.1,0.65,0.03])
    axkernel = plt.axes([0.25,0.15,0.65,0.03])

    #Creat sliders
    ssigma  = Slider(axsigma,'Sigma',0,20,valinit=0,valstep=0.1)
    skernel  = Slider(axkernel,'Kernel',1,10,valinit=0,valstep=1)

    def update(val):
        kernel = int(skernel.val * 2 + 1)
        sigma  = ssigma.val

        gauss_img = cv2.GaussianBlur(img,(kernel,kernel),sigma)
        fig.set_data(np.hstack((img,gauss_img)))

    ssigma.on_changed(update)
    skernel.on_changed(update)

    plt.show()

#Excercise 3 - Median filtering
def excercise_3(img_path):
    #cv2.medianBlur(src,ksize) -> dst

    ksize = 3 #Should be odd integer > 1

    #read in black and white image
    img    = cv2.imread(img_path,0)
    median = cv2.medianBlur(img,ksize)

    plt.figure("Median Blur")
    pltImg = plt.imshow(np.hstack((img,median)),cmap='gray')

    sksize  = Slider(plt.axes([0.15,0.1,0.65,0.03]),'ksize',1,25,valinit=0,valstep=1)
    def update(val):
        ksize = int(sksize.val * 2 + 1)
        median = cv2.medianBlur(img,ksize)
        pltImg.set_data(np.hstack((img,median)))

    sksize.on_changed(update)
    plt.show()

#Excercise 4 - Pixel transform


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
        func = [excercise_1,excercise_2,excercise_3]
        func[int(sys.argv[2])-1](sys.argv[1])
