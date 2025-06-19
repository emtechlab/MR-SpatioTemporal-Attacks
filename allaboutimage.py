# import the necessary packages
from matplotlib.cbook import is_math_text
import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt
import imutils
from PIL import Image

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged

def hist_plot(img,m,n):
    # calculate mean value from RGB channels and flatten to 1D array
    vals = image.mean(axis=2).flatten()
    # plot histogram with 255 bins
    b, bins, patches = plt.hist(vals, 255)
    # print(b, len(bins), len(patches))
    plt.xlim([0,255])
    plt.show()
    return (bins, patches)


def plot_histogram(image, title, mask=None):
    # split the image into its respective channels, then initialize
    # the tuple of channel names along with our figure for plotting
    chans = cv2.split(image)
    colors = ("b", "g", "r")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    # loop over the image channels
    for (chan, color) in zip(chans, colors):
        # create a histogram for the current channel and plot it
        hist = cv2.calcHist([chan], [0], mask, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    # M = np.float32([[1,0,tx],[0,1,ty]])
    # Negative values for the t_{x} value will shift the image to the left
    # Positive values for t_{x} shifts the image to the right
    # Negative values for t_{y} shifts the image up
    # Positive values for t_{y} will shift the image down

if __name__ == "__main__":
    input_path = "/home/.../PV/"
    imagelist = [f for f in glob.glob(input_path + "*.png")]
    imagelist = sorted(imagelist)
    size = 1
    # loop over the images
    for i in range(size):
        image = cv2.imread(str(imagelist[1000]))
        image2 = cv2.imread(str(imagelist[1001]))
        frame_diff = cv2.absdiff(image, image2)
        # Stack the 3 images into a 4d sequence
        sequence = np.stack((image, image, frame_diff), axis=3)

        # Repace each pixel by mean of the sequence
        result = np.median(sequence, axis=3).astype(np.uint8)
        frame_diff = cv2.absdiff(image, image2)
        # convert to graky
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # threshold input image as mask
        mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)[1]
        # print(mask)

        # negate mask
        mask = 255 - mask

        # apply morphology to remove isolated extraneous noise
        # use borderconstant of black since foreground touches the edges
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # anti-alias the mask -- blur then stretch
        # blur alpha channel
        mask = cv2.GaussianBlur(mask, (0,0), sigmaX=2, sigmaY=2, borderType = cv2.BORDER_DEFAULT)

        # linear stretch so that 127.5 goes to 0, but 255 stays 255
        mask = (2*(mask.astype(np.float32))-255.0).clip(0,255).astype(np.uint8)

        # put mask into alpha channel
        result = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        result[:, :, 3] = mask
        # cv2.imshow("backgroun", result)
        # Calculate the histograms, and normalize them
        hist_img1 = cv2.calcHist([image], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist_img1, hist_img1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        hist_img2 = cv2.calcHist([image2], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist_img2, hist_img2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        diff = 255 - cv2.absdiff(image, image2)
        edges_rgb = cv2.cvtColor(auto_canny(diff), cv2.COLOR_GRAY2RGB)
        dst = cv2.addWeighted(image,1.0,edges_rgb,1.0,0)
        cv2.imshow('diff', diff)
        cv2.imshow('new', dst)
        cv2.imshow('image1', image)
        cv2.imshow('image2', image2)
        # Find the metric value
        metric_val = cv2.compareHist(hist_img1, hist_img2, cv2.HISTCMP_CORREL)
        print(metric_val)
        edges_rgb = cv2.cvtColor(auto_canny(image), cv2.COLOR_GRAY2RGB)
        dst = cv2.addWeighted(image,1.0,edges_rgb,1.0,0)
        hist_img2 = cv2.calcHist([dst], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist_img2, hist_img2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        # Find the metric value
        metric_val = cv2.compareHist(hist_img1, hist_img2, cv2.HISTCMP_CORREL)
        print(metric_val)
        # cv2.imshow("overlay", dst)
        cv2.waitKey(0)
        # To ascertain total numbers of rows and 
    # columns of the image, size of the image
        h, w, c = image.shape
        # m, n = image.shape
        r1, count1 = hist_plot(image, h,w)
        
        # plotting the histogram
        # plt.stem(r1, count1)
        plt.xlabel('intensity value')
        plt.ylabel('number of pixels')
        plt.title('Histogram of the original image')
        
        # Transformation to obtain stretching
        constant = (255-0)/(image.max()-image.min())
        img_stretch = image * constant
        # r, count = hist_plot(img_stretch)
        
        # # plotting the histogram
        # plt.stem(r, count)
        plt.xlabel('intensity value')
        plt.ylabel('number of pixels')
        plt.title('Histogram of the stretched image')
        
        # Storing stretched Image
        # cv2.imwrite('Stretched Image 4.png', img_stretch)
        M = np.float32([[1,0,25],[0,1,50]])
        
        # Define the 3 pairs of corresponding points
        input_pts = np.float32([[0,0], [h-1,0], [0,w-1]])
        output_pts = np.float32([[h-1,0], [0,0], [h-1,w-1]])
        shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        matrix = cv2.getAffineTransform(input_pts, output_pts)
        # Apply the affine transformation using cv2.warpAffine()
        dst = cv2.warpAffine(image, matrix, (h,w))
        flipped = cv2.flip(image, 1)
        # cv2.imshow('flipped', flipped)
        hist = cv2.calcHist(cv2.split(flipped), [0], None, [256], [0, 256])
        shifted = imutils.translate(image, 0, 100)
        # cv2.imshow("Shifted Down", shifted)
        # cv2.waitKey(0)
        colors = ["b","g","r"]
        # hist = cv2.calcHist(cv2.split(image), [0], None, [256], [0, 256])
        fig = plt.figure()
        ax = plt.axes()
        # plt.plot(hist)
        plt.xlim([0, 256])
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        # apply Canny edge detection using a wide threshold, tight
        # threshold, and automatically determined threshold
        # wide = cv2.Canny(blurred, 10, 200)
        # tight = cv2.Canny(blurred, 225, 250)
        auto = auto_canny(image)
    
