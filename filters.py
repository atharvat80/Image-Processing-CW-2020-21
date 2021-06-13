"""
Author      : htrf88
Description : Filters
"""

import cv2
import math
import numpy as np
from colorsys import hls_to_rgb
from scipy.interpolate import UnivariateSpline


# Display image
# If you would like to save the processed images set save to True
def show(title, img, save=False):
    cv2.imshow(title, img)
    cv2.waitKey()

    if save:
        cv2.imwrite(title + '.jpg', img)


"""
------------------------------------------------------------------------------------
   Problem 1
------------------------------------------------------------------------------------
"""
# Generate array of specified length with colours of the raindow
def getRainbow(width, end=2/3):
    rnb_float = [hls_to_rgb(end * i/(width-1), 0.5, 1) for i in range(width)]
    rnb = np.array([[255*i[0], 255*i[1], 255*i[2]] for i in rnb_float])
    return rnb


# Generate mask of size image.shape
def getMask(rows, cols, beamWidth=40, rainbow=0):
    mask = np.zeros((rows, cols, 3))
    start, end = int(cols*0.35), int(cols*0.35) + beamWidth
    sub = []

    if rainbow:
        sub = getRainbow(beamWidth)
    else:
        sin = 255*np.sin(np.linspace(0, np.pi, beamWidth))
        sub = np.stack((sin, sin, sin), axis=1)

    step, shift = 5, 0
    for i in range(0, rows, step):
        for j in range(step):
            try:
                mask[i+j, (start+shift):(end+shift)] = sub
            except:
                pass
        shift += 1

    mask = mask.astype('uint8')
    mask = cv2.GaussianBlur(mask, (9, 9), 0)
    return mask


"""

Problem 1 Parameters

img     : Input image as a numpy array
darkCf  : Darkening coefficient, float in range 0 to 1
blendCf : Blending coefficient, float in range 0 to 1
mode    : Type of light leak, string, "simple" or "rainbow"

"""
def problem1(img, darkCf=0.1, blendCf=0.3, mode="simple"):
    width, height, ch = img.shape
    mask = []

    # Generate Mask
    if mode == "rainbow":
        mask = getMask(width, height, beamWidth=40, rainbow=1)
        show("Generated mask", mask)
    else:
        mask = getMask(width, height, beamWidth=60)
        show("Generated mask", mask)

    # Blend mask and image
    img = img * (1 - darkCf)
    show("Darkened image", img.astype('uint8'))
    final = img * (1 - blendCf) + mask * blendCf
    final = final.astype('uint8')
    show("Blended image and mask", final)


"""
------------------------------------------------------------------------------------
   Problem 2
------------------------------------------------------------------------------------
"""
# bilinear interpolation on the pint x,y in mat
def smoothNoise(x, y, mat):
    width, height = mat.shape
    fracX = x - int(x)
    fracY = y - int(y)

    x1 = (int(x) + width) % width
    y1 = (int(y) + height) % height
    x2 = (x1 + width - 1) % width
    y2 = (y1 + height - 1) % height

    value = 0.0
    value += fracX * fracY * mat[y1][x1]
    value += (1 - fracX) * fracY * mat[y1][x2]
    value += fracX * (1 - fracY) * mat[y2][x1]
    value += (1 - fracX) * (1 - fracY) * mat[y2][x2]

    return value


# Add motion blur to img
def motionBlur(img, size=7, dir='l'):
    kernel = np.identity(size)
    if dir == 'r':
        kernel = np.flip(kernel, axis=1)
    elif dir == 'h':
        kernel = np.zeros((size, size))
        kernel[size//2] = np.ones(size)
    
    kernel /= size
    return cv2.filter2D(img, -1, kernel)


# Add Gaussian noise to the img
def addNoise(img, sigma=10):
    width, height = img.shape
    mean = 0
    gauss = np.random.normal(mean, sigma, (width, height))
    gauss = gauss.reshape(width, height)
    noisy = gauss + img
    return noisy.clip(0, 255).astype('uint8')


# Generate texture
def texture(width, height, size, blur, dir='l'):
    blank = np.full((width, height), 209)
    noisy = addNoise(blank, sigma=20)
    img = np.zeros((width, height))
    if size > 1:
        for x in range(width):
            for y in range(height):
                img[x, y] = smoothNoise(x/size, y/size, noisy)
    else:
        img = noisy
    
    if blur > 0:
        img = motionBlur(img, blur, dir)
    
    return img.astype('uint8')


"""

Problem 2 Parameters

img         : Input image as a numpy array
blendCf     : Blending coefficient, float in range 0 to 1
strokeLen   : Length of the pencil stroke, a positive non-zero integer
mode        : Type of light leak, string, "simple" or "colour"
hex         : Hex value of the desired colour for the output sketch in colour mode

"""
def problem2(img, blendCF=0.3, strokeLen=7, mode="simple", hex="ffa600"):
    # Convert to gray
    gray = 0.114*img[:, :, 0] + 0.587 * img[:, :, 1] + 0.2989*img[:, :, 2]
    gray = gray.astype('uint8')
    show("Gray image", gray)

    # Generate sketch
    blurred = cv2.GaussianBlur(gray, (75, 75), 0, 0)
    sketch = cv2.divide(gray, blurred, scale=255)
    show("Sketch (without texture)", sketch)
    sketch = 255 * (sketch/255)**3 # Increasing the intensity here as it will be reduced in blending 

    # Generate texture
    width, height = gray.shape
    strokeLen = strokeLen if strokeLen % 2 == 1 else strokeLen + 1
    paper_1 = texture(width, height, 2, strokeLen)
    show("Texture 1", paper_1)

    # Combine texture and sketch
    sketch_txr_1 = sketch * (1 - blendCF) + paper_1 * blendCF
    sketch_txr_1 = sketch_txr_1.astype('uint8')

    # Display image
    if mode == "colour":
        show("Texture 1 + sketch", sketch_txr_1)
        # Generate 2 more textures
        paper_2 = texture(width, height, 1.8, strokeLen+3, dir='r')
        show("Texture 2", paper_2)
        sketch_txr_2 = sketch * (1 - blendCF) + paper_2 * blendCF
        sketch_txr_2 = sketch_txr_2.astype('uint8')
        show("Texture 2 + sketch", sketch_txr_2)
        
        paper_3 = texture(width, height, 1.2, 3, dir='h')
        show("Texture 3", paper_3)  
        sketch_txr_3 = sketch * (1 - blendCF) + paper_3 * blendCF
        sketch_txr_3 = sketch_txr_3.astype('uint8')
        show("Texture 3 + sketch", sketch_txr_3)

        # Empty array to store the result image
        colour_sketch = np.zeros((width, height, 3))
        colour = [int(hex[i:i+2], 16) for i in (0, 2, 4)]
        for x in range(width):
            for y in range(height):
                colour_sketch[x, y, 0] = sketch_txr_1[x, y]/255 * colour[2]
                colour_sketch[x, y, 1] = sketch_txr_2[x, y]/255 * colour[1]
                colour_sketch[x, y, 2] = sketch_txr_3[x, y]/255 * colour[0]

        colour_sketch = colour_sketch.astype('uint8')
        show("Final coloured Sketch", colour_sketch)
    else:
        show("Final sketch", sketch_txr_1)


"""
------------------------------------------------------------------------------------
   Problem 3
------------------------------------------------------------------------------------
"""
# Median filter (1 channel)
def medianFilter(img, k):
    width, height = img.shape
    output = img.copy()
    window = np.zeros(k * k)
    edge = math.floor(k / 2)
    for x in range(edge, width - edge):
        for y in range(edge, height - edge):
            i = 0
            for fx in range(k):
                for fy in range(k):
                    window[i] = img[x + fx - edge, y + fy - edge]
                    i = i + 1
            window = np.sort(window)
            output[x, y] = window[(k * k) // 2]

    return output.astype('uint8')


# Median filter (for 3 channels)
def median3D(img, k):
    b, g, r = cv2.split(img)
    b = medianFilter(b, k)
    g = medianFilter(g, k)
    r = medianFilter(r, k)
    return cv2.merge((b, g, r))


def createLUT(x, y):
    spl = UnivariateSpline(x, y)
    return spl(np.arange(0, 256))


def curveFilter(img):
    blue, green, red = cv2.split(img)

    increase = createLUT([0, 64, 128, 192, 256], [0, 75, 145, 215, 256])
    increase_slow = createLUT([0, 64, 128, 192, 256], [0, 60, 120, 190, 256])

    # Apply colour curve
    red = cv2.LUT(red, increase).astype('uint8')
    blue = cv2.LUT(blue, increase_slow).astype('uint8')

    # Plot colour curve of the image
    # rng = np.arange(0,256)
    # plt.plot(rng, rng, 'green')
    # plt.plot(rng, increase, 'red')
    # plt.plot(rng, increase_slow, 'blue')
    # plt.show()

    return cv2.merge((blue, green, red))


def CLAHE(img, clipLim=0.8, tileSize=5):
    blue, green, red = cv2.split(img)

    clahe = cv2.createCLAHE(
        clipLimit=clipLim, tileGridSize=(tileSize, tileSize))

    blue = clahe.apply(blue)
    green = clahe.apply(green)
    red = clahe.apply(red)

    return cv2.merge((blue, green, red))


"""

Problem 3 Parameters

img     : Input image as a numpy array
blur    : Amount of blur to be applied to the input image, a positive non-zero integer

"""
def problem3(img, blur=3):
    filtered = median3D(img, blur)
    equalised = CLAHE(filtered)
    curved = curveFilter(equalised)
    show("After applying median blur", filtered)
    show("After applying CLAHE", equalised)
    show("A hint of Autumn", curved)


"""
------------------------------------------------------------------------------------
   Problem 4
------------------------------------------------------------------------------------
"""
# Low pass filter using Box blur


def lowPassFilter(img):
    kernel = np.ones((3, 3))
    kernel = kernel/9
    return cv2.filter2D(img, -1, kernel)


# Apply swirl transform
def swirl(img, strength, radius, flag="swirl"):
    height, width = img.shape[:2]
    cx, cy = width//2 - 1, height//2 - 1
    maxRadius = min(cx, cy) + 1
    if radius > maxRadius or radius <= 0:
        radius = maxRadius

    result = img.copy()
    for x in range(width):
        for y in range(height):
            # Calculate the current points distance from the center (cx, cy)
            dx = x - cx
            dy = y - cy
            dist = math.sqrt((dx * dx) + (dy * dy))
            theta = math.atan2(dy, dx)
            swirl_amount = 1 - (dist / radius)

            # Determine if the point is within specified radius of center (cx, cy)
            if swirl_amount > 0:
                # Calculate the position of pixel (x,y) after applying the transform
                swirl_amount = 0.1*(10**swirl_amount) - 0.1
                if flag == "swirl":
                    theta -= swirl_amount * math.pi * strength
                else:
                    theta += swirl_amount * math.pi * strength  # unswirl

                dx = math.cos(theta) * dist
                dy = math.sin(theta) * dist
                xi, yi = cx + dx, cy + dy

                # Bilinear interpolation
                x1, x2 = math.floor(cx + dx), math.ceil(cx + dx)
                y1, y2 = math.floor(cy + dy), math.ceil(cy + dy)
                a, b = x2-x1, y2-y1
                if a and b:
                    R1 = ((x2-xi)/a)*img[x1, y1] + ((xi-x1)/a)*img[x2, y1]
                    R2 = ((x2-xi)/a)*img[x1, y2] + ((xi-x1)/a)*img[x2, y2]
                    P = ((y2-yi)/b) * R1 + ((yi-y1)/b) * R2
                else:
                    P = img[int(xi), int(yi)]
                result[x, y] = P.astype(int)
            else:
                result[x, y] = img[x, y]

    return result


"""

Problem 4 Parameters

img         : Input image as a numpy array
strength    : Strength of the swirl, float 
radius      : Swirl radius, integer

"""
def problem4(img, strength=0.9, radius=200):
    filtered = lowPassFilter(img)
    filtered_swirled = swirl(filtered, strength, radius)
    swirled = swirl(img, strength, radius)
    unswirled = swirl(swirled, strength, radius, 'unswirl')

    show("Low pass filter applied", filtered)
    show("Low pass filter applied and swirled", filtered_swirled)
    show("Swirled (without low pass filter)", swirled)
    show("Previous image unswirled", unswirled)
    show("Difference between the original and unswirled", img - unswirled)


"""
------------------------------------------------------------------------------------
   Run the program here
------------------------------------------------------------------------------------
"""

if __name__ == "__main__":
    
    filename = 'face1.jpg'
    img = cv2.imread(filename)
    
    if img is not None:
        # call the desired function here, some examples below
        # problem1(img, darkCf=0.15, blendCf=0.35, mode="rainbow")
        # problem2(img, blendCF=0.3, strokeLen=9, mode="colour")
        # problem3(img, 3)
        # problem4(img, strength=1.2, radius=150)
        pass
    else:
        print(filename, "not found.")