import numpy as np 
import imageio # na ziskanie obr
import matplotlib.pyplot as plt 

#------------------------

#funkcia na ziskanie matice LOG
def logFilter(size, sigma):

    x=np.ones((5,1))*np.array(range(-int(np.floor(size / 2)), int(np.ceil(size / 2))))
    y=x.T
   
    k = np.exp(-(x*x+y*y)/(2*sigma**2)) / np.sum(np.exp(-(x*x+y*y)/(2*sigma**2)))

    laplace = k*(x*x+y*y - 2*sigma**2)/(2*np.pi*sigma**6) - np.sum(k*(x*x+y*y - 2*sigma**2)/(2*np.pi*sigma**6))/(size*size)
    return laplace
    
#pouzitie LOG matici v konvolucii
def LoG(source):
    img = imageio.imread('lenna.png', pilmode='L') 
    
    height = img.shape[0]
    width = img.shape[1]
    imgOut=np.zeros((height,width),dtype=np.uint8) #zaplnime 0mi

    laplace = logFilter(5, 1)

    # Convolution: Zacnem 2,2 mieste
    center = 2
    for i in np.arange(center, height-center): 
        for j in np.arange(center, width-center):        
            sum = 0 
            for kernel_i in np.arange(-center, 3):  # jednolive hodnoti kernela nasobime s prislusnzmi castami obr
                for kernel_j in np.arange(-center, 3): 
                    value = img.item(i+kernel_i, j+kernel_j) # ziskame hodnotu pixelu
                    weight = laplace[center+kernel_i, 2+kernel_j] #ziskame hodnotu kernelu       
                    sum = sum + (weight * value) # vykoname sumaciu
            kernelOut = sum
            imgOut.itemset((i,j), kernelOut) # priradime vysledok ku konkretnemu pixelu
    return [img,imgOut]
  
img,img_out=LoG('lenna.png')
plt.imshow(img,cmap=plt.get_cmap('gray')) 
plt.imshow(img_out,cmap=plt.get_cmap('gray')) 
#-----------------------------
plt.imshow(img,cmap=plt.get_cmap('gray')) 
  
