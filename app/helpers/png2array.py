from PIL import Image
import numpy as np
import glob, os

def image2pixelarray(filepath):
    """
    Parameters
    ----------
    filepath : str
        Path to an image file

    Returns
    -------
    list
        A list of lists which make it simple to access the greyscale value by
        im[y][x]
    """
    im = Image.open(filepath).convert('L')
    im = im.resize(64,64)
    (width, height) = im.size; print(im.size)
    greyscale_map = list(im.getdata())
    greyscale_map = np.array(greyscale_map)
    greyscale_map = greyscale_map.reshape((height, width))
    return greyscale_map #.resize(64,64)


paths = list(sorted(glob.glob("/home/marten/Desktop/workdir/gans/app-dev/notebooks/2*/*/*.gif")))
for filename in paths:
	try:
		img = Image.open(filename)  # open the image file
		img.verify()  # verify that it is, in fact an image
		print('OK file:', filename)
		array = image2pixelarray(filename)
		np.save(
			filename.replace('.gif',''), array
		)
		print('--'*10)
	except (IOError, SyntaxError) as e:
		print('Bad file:', filename)  # print out the names of corrupt files