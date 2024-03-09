import cv2

radar_image = cv2.imread('input_image', cv2.IMREAD_GRAYSCALE)

window_size = 20

filtered_image = cv2.fastNlMeansDenoising(radar_image, None, h=window_size)

cv2.imshow('Original Image', radar_image)
cv2.imshow('Filtered Image', filtered_image)
cv2.waitKey(0)
