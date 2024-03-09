import cv2

image_path = "path_to_input_image"
output_folder = "path_to_output_folder"
window_size = 20
def filter(image_path, output_folder, window_size):
    radar_image = cv2.imread('path_to_input_image')

    filtered_image = cv2.fastNlMeansDenoising(radar_image, None, h=window_size)

    #cv2.imshow('Original Image', radar_image)
    #cv2.imshow('Filtered Image', filtered_image)
    #cv2.waitKey(0)

    output_path = output_folder + "/filtered_image.jpg"
    cv2.imwrite(output_path, filtered_image)