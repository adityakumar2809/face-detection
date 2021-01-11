import cv2


image_index = 0


def getImage():
    img = cv2.imread('images/image1.jpg')
    return img


def showImage(img):
    global image_index
    cv2.imshow(f'Image#{image_index}', img)
    image_index += 1


def getGrayscaleImage(img):
    gray_image = cv2.cvtColor(
        src=img,
        code=cv2.COLOR_BGR2GRAY
    )
    return gray_image


def getCascadeClassifier():
    haar_cascade = cv2.CascadeClassifier('haar_face.xml')
    return haar_cascade


def detectFace(cascade_classifier, img):
    faces_rect = cascade_classifier.detectMultiScale(
        image=img,
        scaleFactor=1.1,
        minNeighbors=3
    )
    return faces_rect


def main():
    img = getImage()
    showImage(img)

    gray_image = getGrayscaleImage(img)
    showImage(gray_image)

    haar_cascade = getCascadeClassifier()

    faces_rect = detectFace(haar_cascade, gray_image)
    print(f'Number of faces found is {len(faces_rect)}')

    cv2.waitKey(0)


if __name__ == "__main__":
    main()
