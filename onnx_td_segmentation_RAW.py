# me - this DAT
# scriptOp - the OP which is cooking

import time

import numpy as np
import onnxruntime as onnxruntime

import cv2
import numpy as np

import math
from colorsys import hls_to_rgb

# Array of YOLOv8 class labels
yolo_classes = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
    "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Create a list of colors for each class where each color is a tuple of 3 integer values
rng = np.random.default_rng(3)
colors = rng.uniform(0, 255, size=(len(yolo_classes), 3))

# Create a smooth gradient of colors using HSL color space
hue_values = np.linspace(0, 0.5, len(yolo_classes))
colors = [hls_to_rgb(hue, 0.5, 1) for hue in hue_values]

# Convert RGB values to integers in the range [0, 255]
colors = (np.array(colors) * 255)


# press 'Setup Parameters' in the OP to call this function to re-create the parameters.
def onSetupParameters(scriptOp):
    page = scriptOp.appendCustomPage('Fast Neural Style')
    p = page.appendFile('Onnxmodel', label='ONNX Model')
    return

# called whenever custom pulse parameter is pushed
def onPulse(par):
    return

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

Modelpath = str(op('script2').par.Onnxmodel)


def onCook(scriptOp):
    print("Start")
    img = scriptOp.inputs[0].numpyArray()
    img_scaled = (img[:, :, :3] * 255).astype(np.uint8)

    # Prepare input for YOLOv8
    input_tensor, img_width, img_height = prepare_input_from_numpy(img_scaled)
    print(input_tensor.shape, img_width, img_height)

    # Run YOLOv8 model
    outputs = run_yolov8_model(input_tensor)
    output0 = outputs[0]
    output1 = outputs[1]
    print("Output0:",output0.shape,"Output1:",output1.shape)

    # Process YOLOv8 output
    detected_objects = process_output(outputs, img_width, img_height)
    print(detected_objects)
    # Visualization
    combined_img = img_scaled.copy()

    for obj in detected_objects:
        x1, y1, x2, y2, label, prob, polygon = obj

        # Draw bounding box
        color = colors[yolo_classes.index(label)]
        cv2.rectangle(combined_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        # Draw label and probability
        label_text = f'{label} {prob:.2f}'
        cv2.putText(combined_img, label_text, (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw polygon (optional)
        cv2.polylines(combined_img, [np.array(polygon)], isClosed=True, color=color, thickness=2)

    # Update the output image
    scriptOp.copyNumpyArray(combined_img)

    return


def prepare_input_from_numpy(img_scaled):
    # Resize the image to (640, 640)
    img_resized = cv2.resize(img_scaled, (640, 640))

    # Convert the image to RGB format
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    # Normalize the pixel values to the range [0, 1]
    input_tensor = img_rgb.astype(np.float32) / 255.0

    # Transpose the dimensions to (3, 640, 640) and add a batch dimension
    input_tensor = np.transpose(input_tensor, (2, 0, 1)).reshape(1, 3, 640, 640)

    # Get the dimensions of the original image
    img_height, img_width, _ = img_scaled.shape

    return input_tensor.astype(np.float32), img_width, img_height



def run_yolov8_model(input_tensor):
    model = onnxruntime.InferenceSession(Modelpath)
    outputs = model.run(None, {"images": input_tensor})
    return outputs


def process_output(outputs, img_width, img_height):
    """
    Function used to convert RAW output from YOLOv8 to an array
    of detected objects. Each object contain the bounding box of
    this object, the type of object and the probability
    :param outputs: Raw outputs of YOLOv8 network
    :param img_width: The width of original image
    :param img_height: The height of original image
    :return: Array of detected objects in a format [[x1,y1,x2,y2,object_type,probability],..]
    """
    output0 = outputs[0].astype("float")
    output1 = outputs[1].astype("float")
    output0 = output0[0].transpose()
    output1 = output1[0]
    boxes = output0[:, 0:84]
    masks = output0[:, 84:]
    output1 = output1.reshape(32, 160 * 160)
    masks = masks @ output1
    boxes = np.hstack((boxes, masks))
    print("boxes",len(boxes))
    objects = []
    counter = 0
    for row in boxes:
        prob = row[4:84].max()
        print(prob)
        if prob < 0.005:
            continue

        class_id = row[4:84].argmax()
        label = yolo_classes[class_id]
        xc, yc, w, h = row[:4]
        x1 = (xc - w/2) / 640 * img_width
        y1 = (yc - h/2) / 640 * img_height
        x2 = (xc + w/2) / 640 * img_width
        y2 = (yc + h/2) / 640 * img_height
        mask = get_mask(row[84:25684], (x1, y1, x2, y2), img_width, img_height)
        polygon = get_polygon(mask)
        objects.append([x1, y1, x2, y2, label, prob, polygon])

    objects.sort(key=lambda x: x[5], reverse=True)
    result = []
    while len(objects) > 0:
        result.append(objects[0])
        objects = [object for object in objects if iou(object, objects[0]) < 0.5]

    print(counter)
    return result


def get_mask_OLD(row,box, img_width, img_height):
    """
    Function extracts segmentation mask for object in a row
    :param row: Row with object
    :param box: Bounding box of the object [x1,y1,x2,y2]
    :param img_width: Width of original image
    :param img_height: Height of original image
    :return: Segmentation mask as NumPy array
    """
    mask = row.reshape(160,160)
    mask = sigmoid(mask)
    mask = (mask > 0.5).astype('uint8')*255
    x1,y1,x2,y2 = box
    mask_x1 = round(x1/img_width*160)
    mask_y1 = round(y1/img_height*160)
    mask_x2 = round(x2/img_width*160)
    mask_y2 = round(y2/img_height*160)
    mask = mask[mask_y1:mask_y2,mask_x1:mask_x2]
    img_mask = Image.fromarray(mask,"L")
    img_mask = img_mask.resize((round(x2-x1),round(y2-y1)))
    mask = np.array(img_mask)
    return mask

def get_mask(row, box, img_width, img_height):
    """
    Function extracts segmentation mask for object in a row
    :param row: Row with object
    :param box: Bounding box of the object [x1, y1, x2, y2]
    :param img_width: Width of the original image
    :param img_height: Height of the original image
    :return: Segmentation mask as NumPy array
    """
    mask = row.reshape(160, 160)
    mask = sigmoid(mask)
    mask = (mask > 0.5).astype('uint8') * 255
    x1, y1, x2, y2 = box
    mask_x1 = round(x1 / img_width * 160)
    mask_y1 = round(y1 / img_height * 160)
    mask_x2 = round(x2 / img_width * 160)
    mask_y2 = round(y2 / img_height * 160)
    mask = mask[mask_y1:mask_y2, mask_x1:mask_x2]

    # Resize the mask using OpenCV
    mask = cv2.resize(mask, (round(x2 - x1), round(y2 - y1)))

    return mask

def get_polygon(mask):
    """
    Function calculates bounding polygon based on segmentation mask
    :param mask: Segmentation mask as Numpy Array
    :return:
    """
    contours = cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    polygon = [[int(contour[0][0]), int(contour[0][1])] for contour in contours[0][0]]
    return polygon

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def iou(box1,box2):
    """
    Function calculates "Intersection-over-union" coefficient for specified two boxes
    https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/.
    :param box1: First box in format: [x1,y1,x2,y2,object_class,probability]
    :param box2: Second box in format: [x1,y1,x2,y2,object_class,probability]
    :return: Intersection over union ratio as a float number
    """
    return intersection(box1,box2)/union(box1,box2)


def union(box1,box2):
    """
    Function calculates union area of two boxes
    :param box1: First box in format [x1,y1,x2,y2,object_class,probability]
    :param box2: Second box in format [x1,y1,x2,y2,object_class,probability]
    :return: Area of the boxes union as a float number
    """
    box1_x1,box1_y1,box1_x2,box1_y2 = box1[:4]
    box2_x1,box2_y1,box2_x2,box2_y2 = box2[:4]
    box1_area = (box1_x2-box1_x1)*(box1_y2-box1_y1)
    box2_area = (box2_x2-box2_x1)*(box2_y2-box2_y1)
    return box1_area + box2_area - intersection(box1,box2)


def intersection(box1,box2):
    """
    Function calculates intersection area of two boxes
    :param box1: First box in format [x1,y1,x2,y2,object_class,probability]
    :param box2: Second box in format [x1,y1,x2,y2,object_class,probability]
    :return: Area of intersection of the boxes as a float number
    """
    box1_x1,box1_y1,box1_x2,box1_y2 = box1[:4]
    box2_x1,box2_y1,box2_x2,box2_y2 = box2[:4]
    x1 = max(box1_x1,box2_x1)
    y1 = max(box1_y1,box2_y1)
    x2 = min(box1_x2,box2_x2)
    y2 = min(box1_y2,box2_y2)
    return (x2-x1)*(y2-y1)
