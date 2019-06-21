#!/usr/bin/python
import numpy as np
import cv2
import os
import argparse

def createDirectories():
    if not os.path.exists('ImagesGrayscale'):
        os.makedirs('ImagesGrayscale')

    if not os.path.exists('ImagesDarkened'):
        os.makedirs('ImagesDarkened')

    if not os.path.exists('ImagesDarkenedColor'):
            os.makedirs('ImagesDarkenedColor')

    if not os.path.exists('ImagesFlippedHorizontally'):
            os.makedirs('ImagesFlippedHorizontally')
            
    if not os.path.exists('ImagesFlippedVertically'):
            os.makedirs('ImagesFlippedVertically')

    if not os.path.exists('ImagesFlippedHorizontallyAndVertically'):
            os.makedirs('ImagesFlippedHorizontallyAndVertically')

    if not os.path.exists('ImagesFlippedNoisy'):
            os.makedirs('ImagesFlippedNoisy')

    if not os.path.exists('ImagesNoisyDark'):
            os.makedirs('ImagesNoisyDark')

    if not os.path.exists('ImagesBlurred'):
            os.makedirs('ImagesBlurred')
            
def initArgs():
    parser = argparse.ArgumentParser(description='Python Program that implements image augmentation for deep learning datasets.')
    parser.add_argument('--dir', nargs='?', const='Images', type=str, default='Images', metavar='DIR', help='The directory where you stored your images')
    args = parser.parse_args()
    dir = args.dir
    return dir

def grayscale(img):
    currentImg = cv2.imread(img, 0)
    cv2.imwrite('./ImagesGrayscale/gray_' + img, currentImg)

def darken(img):
    currentImg = cv2.imread(img, 0)
    darkened = currentImg * 0.4
    cv2.imwrite('./ImagesDarkened/dark_' + img, darkened)

def darkenColored(img):
    currentImg = cv2.imread(img, 1)
    darkenedColor = currentImg * 0.4
    cv2.imwrite('./ImagesDarkenedColor/darkCol_' + img, darkenedColor)
    
def flipHorizontally(img):
    currentImg = cv2.imread(img, 0)
    flippedHorizontally = np.flipud(currentImg)
    cv2.imwrite('./ImagesFlippedHorizontally/flipHoriz_' + img, flippedHorizontally)

def flipVertically(img):
    currentImg = cv2.imread(img, 0)
    flippedVertically = np.fliplr(currentImg)
    cv2.imwrite('./ImagesFlippedVertically/flipVert_' + img, flippedVertically)

def flipHorizontallyAndVertically(img):
    currentImg = cv2.imread(img, 0)
    flippedVertically = np.fliplr(currentImg)
    flippedHorizontallyAndVertically = np.flipud(flippedVertically)
    cv2.imwrite('./ImagesFlippedHorizontallyAndVertically/flipHorizVert_' + img, flippedHorizontallyAndVertically)

def addNoise(img):
    currentImg = cv2.imread(img, 0)
    width, height = img.shape
    noise = np.zeros((width, height))
    cv2.randu(noise, 0, 256)
    noiseStrength = 0.4
    noisy = currentImg + np.array(noiseStrength*noise, dtype=np.int)
    cv2.imwrite('./ImagesNoisy/noise_' + img, noisy)

def addNoiseDark(img):
    currentImg = cv2.imread(img, 0)
    width, height = img.shape
    noise = np.zeros((width, height))
    cv2.randu(noise, 0, 256)
    noiseStrength = 0.4
    noisy = currentImg + np.array(noiseStrength*noise, dtype=np.int)
    noisyDark = noisy * 0.4
    cv2.imwrite('./ImagesNoisyDark/noiseDark_' + img, noisyDark)

def blur(img, radius):
    currentImg = cv2.imread(img, 0)
    kernelSize = (radius, radius)
    blurry = cv2.blur(currentImg, kernelSize)
    cv2.imwrite('./ImagesBlurred/blur_' + img, blurry)

def main():
    dir = initArgs()
    imgList = os.listdir(dir)
    createDirectories()
    for img in imgList:
        grayscale(img)
        darken(img)
        darkenColored(img)
        flipHorizontally(img)
        flipVertically(img)
        flipHorizontallyAndVertically(img)
        addNoise(img)
        addNoiseDark(img)
        blur(img, 10)

if __name__ == "__main__":
    main()
