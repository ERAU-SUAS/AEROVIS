import os
import random
import cv2 as cv
from src.shape_detector import predict
from src.isolate_character import isolate_character, isolate_character_exp
from src.color_extractor import get_shape_color
import numpy as np
from utils.logger import Logger, LOG_RESULT_IMG_DIR

SAMPLES = 100 
DEFAULT_SHAPE_DETECT_MODEL_PATH = "runs/detect/exp/weights/best.pt"


def add_bound_box(src_img, xywh_tensor_values, label): 
    bbox = []
    for i in range(len(xywh_tensor_values)):
        bbox.append(int(xywh_tensor_values[i]))
    top_left = (bbox[0] - int(bbox[2]/2), bbox[1] - int(bbox[3]/2))
    bottom_right = (top_left[0] + bbox[2], top_left[1] + bbox[3])
    cv.rectangle(src_img, top_left, bottom_right, (0, 255, 0), 1)
    (text_width, text_height), baseline = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    text_origin = (top_left[0], top_left[1] - 10)
    cv.rectangle(src_img, text_origin, (text_origin[0] + text_width, text_origin[1] - text_height - baseline), (0, 255, 0), thickness=cv.FILLED)
    cv.putText(src_img, label, text_origin, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return src_img


def get_random_img():
    dir = os.path.abspath("standard_object_dataset/test/images") 
    files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
    #return "/Users/carson/Documents/Devlopment/SUAS/aero/AEROVIS/standard_object_dataset/test/images/68d5a510f29f5a84c04714f7177e1cb2_V.jpg"
    rfn = random.choice(files)
    s = rfn.split('_')
    return (
        os.path.abspath(f"standard_object_dataset/test/images/{rfn}"),
        s[1],
        s[2],
        s[3]
    )


def crop_image(src_image, xywh):
    for i in range(len(xywh)):
        xywh[i] = int(xywh[i])
    w = xywh[2] - 10 
    h = xywh[3] - 10 
    x = xywh[0] - int(w/2)
    y = xywh[1] - int(h/2) 
    bbox = np.array([x, y, w, h])
    return src_image[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]


def debug():
    img_data = get_random_img()
    random_img_path = img_data[0]
    character = img_data[1]
    shape_color = img_data[2]
    character_color = img_data[3]
    character_color = character_color.split(".")[0]
    
    img = cv.imread(random_img_path)

    res = predict(DEFAULT_SHAPE_DETECT_MODEL_PATH, random_img_path)
    tensor = res[0].boxes.xywh.clone().detach()
    try:
        xywh_tensor_values = tensor.cpu().numpy().tolist()[0] 
    except:
        return
    cropped_img = crop_image(img, xywh_tensor_values)

    random_img_file_name = random_img_path.split('/')[-1]
    og_image_path = f"{LOG_RESULT_IMG_DIR}/{random_img_file_name.split('.')[0]}_cropped.jpg"
    cv.imwrite(og_image_path, cropped_img)

    #isolated_character_image = isolate_character(cropped_img) 
    #isolated_character_image = isolate_character_exp(cropped_img) 

    status, results = isolate_character_exp(cropped_img) 

    for i in range(len(results) + status): # status of -1 -> error, last value in results is the error message 
        result_path = f"{LOG_RESULT_IMG_DIR}/{random_img_file_name.split('.')[0]}_result_{i+1}.jpg"
        cv.imwrite(result_path, results[i])

    if status == -1: 
        Logger.log(character, shape_color, character_color, result_path, og_image_path, error_msg=results[-1]) 
    else:
        Logger.log(character, shape_color, character_color, result_path, og_image_path) 

    #classes = res[0].names
    #box_class = int(res[0].boxes.cls.cpu().numpy().tolist()[0])
    #label = classes[box_class]
    #image_with_box = add_bound_box(img, xywh_tensor_values, label)

    #cv.imshow("shape detection", image_with_box)
    #cv.imshow("isolated character", isolated_character_image)

    #cv.waitKey(0)
    #cv.destroyAllWindows()


def run_debugger(num_samples=SAMPLES):
    for i in range(num_samples):
        debug()
        print(f"[{i+1}/{num_samples}]")
