The functions that correspond to each problem are named as `problem1`, `problem2`, `problem3` and `problem4`. Each function has appropriate defaults set already so all that is required is to pass the image name to be read in using `cv2.imread()` to the functions as an argument.

The default parameters for each function, their datatype, default values and valid range of values are documented below.

# Problem 1
Light leak filter is found in `filters.py` as the function `problem1`
- `img` : Input image as a numpy array
- `darkCf` : Darkening coefficient, float in range 0 to 1, default is 0.1
- `blendCf` : Blending coefficient, float in range 0 to 1, default is 0.3
- `mode` : Type of light leak, string, `simple` or `rainbow`, default is `simple`

# Problem 2
Pencil/Charcoal effect is found in `filters.py` as the function `problem2`
- `img` : Input image as a numpy array
- `blendCf` : Blending coefficient, float in range 0 to 1, default is 0.3
- `strokeLen` : Length of the pencil stroke, a positive non-zero integer, default is 7
- `mode` : Type of light leak, string, `simple` or `colour`, default is `simple`
- `hex` : Hex value of the desired colour for the output sketch in colour mode, default is `ffa600`

# Problem 3
Smoothing and beautifying filter is found in `filters.py` as the function `problem3`

- `img` : Input image as a numpy array
- `blur` : Amount of blur to be applied to the input image, a positive non-zero integer, default is 3

# Problem 4
Face swirl is found in `filters.py` as the function `problem4`
- `img` : Input image as a numpy array
- `strength` : Strength of the swirl, float, default is 0.9 
- `radius` : Swirl radius, integer, default is automatically determined to be the largest possible size.


> Examples of these effects can be found in `report.pdf` along with the details of their implementation.
