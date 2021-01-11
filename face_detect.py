import cv2


image_index = 0


def getImage():
    img = cv2.imread('images/image2.jfif')
    return img


def showImage(img):
    global image_index
    cv2.imshow(f'Image#{image_index}', img)
    image_index += 1


def main():
    img = getImage()
    showImage(img)

    cv2.waitKey(0)


if __name__ == "__main__":
    main()
