import cv2

# 参考：https://blog.csdn.net/luanfenlian0992/article/details/110529737
def fill_hole(gray,threshold):
    ret, img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area <= threshold:
            cv_contours.append(contour)
        else:
            continue

    img = cv2.fillPoly(img, cv_contours, (255, 255, 255))
    return img
def delete_small(gray,threshold):
    ret, img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area <= threshold:
            cv_contours.append(contour)
        else:
            continue

    img = cv2.fillPoly(img, cv_contours, (0, 0, 0))
    return img