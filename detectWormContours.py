import numpy as np
import cv2
import matplotlib
from matplotlib import colors
from matplotlib import pyplot as plt
from os import path
from sys import argv


def getCountours( original_img ):
    """
    Detects contours from a given image and
    saves the resulting image into a new file
    """
    scale = 0.2

    img = cv2.resize( original_img, (0,0), fx=scale, fy=scale)
    image = cv2.cvtColor( img, cv2.COLOR_BGR2RGB)

    # Filters HSV layers
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    images = []
    for i in [0, 1, 2]:
        colour = hsv.copy()
        if i != 0: colour[:,:,0] = 255
        if i != 1: colour[:,:,1] = 255
        if i != 2: colour[:,:,2] = 255
        images.append(colour)

    gray = cv2.bilateralFilter( images[ 1 ], 25, 150, cv2.BORDER_REFLECT )

    edged = cv2.Canny( gray, 30, 175 )

    _, contours, h = cv2.findContours( edged, cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE )

    newContours = []

    for c in contours:
        rect = cv2.minAreaRect( c )
        box = cv2.boxPoints( rect )
        area = cv2.contourArea( box )
        if area > 50:
            newContours.append( c )

    cv2.drawContours( img, newContours, -1, ( 0, 255, 0 ), 2 )

    cv2.imshow( 'Contours', img )
    cv2.waitKey( 0 )

    cv2.destroyAllWindows()

    return img


def saveImageWithContour( image, original_img_name ):
    """
    Saves the image into a new file within the same folder
    """
    name, extension = path.splitext( original_img_name )
    new_image_name = name + "_contours" + extension
    cv2.imwrite( new_image_name, image )


if __name__ == '__main__':
    if len( argv ) < 2:
        print "Usage: %s [FILENAME...]"
    else:
        for file in argv[ 1: ]:
            img = cv2.imread( file )
            final = getCountours( img )
            saveImageWithContour( final, file )
