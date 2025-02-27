
# me - this DAT
# scriptOp - the OP which is cooking

import time

import numpy as np
import onnxruntime as onnxruntime

import cv2
import numpy as np

import math
from colorsys import hls_to_rgb
from PIL import Image, ImageDraw


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

def intersection(box1,box2):
    box1_x1,box1_y1,box1_x2,box1_y2 = box1[:4]
    box2_x1,box2_y1,box2_x2,box2_y2 = box2[:4]
    x1 = max(box1_x1,box2_x1)
    y1 = max(box1_y1,box2_y1)
    x2 = min(box1_x2,box2_x2)
    y2 = min(box1_y2,box2_y2)
    return (x2-x1)*(y2-y1) 

def union(box1,box2):
    box1_x1,box1_y1,box1_x2,box1_y2 = box1[:4]
    box2_x1,box2_y1,box2_x2,box2_y2 = box2[:4]
    box1_area = (box1_x2-box1_x1)*(box1_y2-box1_y1)
    box2_area = (box2_x2-box2_x1)*(box2_y2-box2_y1)
    return box1_area + box2_area - intersection(box1,box2)

def iou(box1,box2):
    iou_res = intersection(box1,box2)/union(box1,box2)
    #print(iou_res)
    return iou_res


def sigmoid(z):
    return 1/(1 + np.exp(-z))

# parse segmentation mask
def get_mask(row, box, img_width, img_height):
    # convert mask to image (matrix of pixels)
    mask = row.reshape(160,160)
    mask = sigmoid(mask)
    mask = (mask > 0.5).astype("uint8")*255
    # crop the object defined by "box" from mask
    x1,y1,x2,y2 = box
    mask_x1 = round(x1/img_width*160)
    mask_y1 = round(y1/img_height*160)
    mask_x2 = round(x2/img_width*160)
    mask_y2 = round(y2/img_height*160)
    mask = mask[mask_y1:mask_y2,mask_x1:mask_x2]
    # resize the cropped mask to the size of object
    img_mask = Image.fromarray(mask,"L")
    img_mask = img_mask.resize((round(x2-x1),round(y2-y1)))
    mask = np.array(img_mask)
    return mask

# calculate bounding polygon from mask


# calculate bounding polygon from mask
def get_polygon_APPROXIMATED(mask, label, prob):
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Assuming there is at least one contour
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Approximate the contour to reduce the number of points
        epsilon = 0.04 * cv2.arcLength(largest_contour, True)
        approx_polygon = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # Convert the polygon to a list of points
        polygon = [[point[0][0], point[0][1]] for point in approx_polygon]
                
        return polygon
    else:
        return None



# calculate bounding polygon from mask
def get_polygon(mask, label, prob):
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Assuming there is at least one contour
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Convert the contour to a list of points
        polygon = [[point[0][0], point[0][1]] for point in largest_contour]
                
        return polygon
    else:
        return None




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

print(" ")
print(" ")
print("model load -x--x-x-x--x-x-x-x-x--x-x-x-x--x-x-x-x-")
model = onnxruntime.InferenceSession(Modelpath, providers=[
    "CUDAExecutionProvider",
    "CPUExecutionProvider"
  ])

 #model = onnxruntime.InferenceSession(Modelpath)

def onCook(scriptOp):

    print("Start")
    
    img = scriptOp.inputs[0].numpyArray()
    img_height, img_width, nchan = img.shape
    ####
    ####
    ####


    # OPEN CV
    img_copy_CV = img[:, :, :3].copy()
    img_copy_CV = np.flip(img_copy_CV, axis=0)

    img_copy_CV = cv2.cvtColor(img_copy_CV, cv2.COLOR_BGR2RGB)
    #print("img_copy_CV second",img_copy_CV)

    img_copy_CV = cv2.resize(img_copy_CV, (640, 640))

    input = img_copy_CV.transpose(2, 0, 1).reshape(1, 3, 640, 640).astype("float32")


    # Run YOLOv8 model
    start_time = time.time()
    print("input secondtry",input[0][0][0][0:2])

    outputs = model.run(None, {"images": input})
    end_time = time.time()

    # Calculate and print execution time in milliseconds
    execution_time = (end_time - start_time) * 1000  # convert seconds to milliseconds
    fps = 1 / (execution_time/1000)
    print(f"Execution time: {execution_time:.2f} ms (FPS): {fps:.2f}")

   
    output0, output1 = outputs
    formatted_output = "out RAW {:.15f}".format(outputs[0][0][0][0])
    print(formatted_output)

    output0 = output0[0].transpose()
    boxes, masks = output0[:, :84], output0[:, 84:]

    output1 = output1.reshape(32, 160 * 160)
    masks = masks @ output1

    boxes = np.hstack([boxes, masks])

    # parse and filter all boxes
    objects = []
    for row in boxes:
        xc,yc,w,h = row[:4]
        x1 = (xc-w/2)/640*img_width
        y1 = (yc-h/2)/640*img_height
        x2 = (xc+w/2)/640*img_width
        y2 = (yc+h/2)/640*img_height

        prob = row[4:84].max()
        if prob < 0.1:
            continue
        class_id = row[4:84].argmax()
        prob = row[5]
        label = yolo_classes[class_id]
        mask = get_mask(row[84:25684], (x1,y1,x2,y2), img_width, img_height)
        polygon = get_polygon(mask,label,prob)
    
        objects.append([x1,y1,x2,y2,label,prob,mask,polygon])

    #print("len objects", len(objects))
    # apply non-maximum suppression
    objects.sort(key=lambda x: x[5], reverse=True)

    result = []

    while len(objects) > 0:
        result.append(objects[0])
        objects = [obj for obj in objects if iou(obj, objects[0]) < 0.5]

    # Create an alpha channel if not present
    if img.shape[2] == 3:
        alpha_channel = np.ones((img.shape[0], img.shape[1]), dtype=img.dtype) * 255
        img = np.dstack((img, alpha_channel))

    for obj in result:
        [x1, y1, x2, y2, label, prob, mask, polygon] = obj
        print(label, "{:.6f}".format(prob), len(polygon))

        # Move polygon from (0,0) to the top left point of the detected object and flip along the Y-axis
        polygon = [(round(x1 + point[0]), round(img.shape[0] - (y1 + point[1]))) for point in polygon]

        # Convert the polygon to numpy array for drawing
        polygon_np = np.array(polygon, dtype=np.int32)

        # Draw the polygon
        cv2.fillPoly(img, [polygon_np], color=(0, 255, 0, 125))

        # Flip and draw the rectangle
        cv2.rectangle(img, (int(x1), int(img.shape[0] - y2)), (int(x2), int(img.shape[0] - y1)),
                    color=(0, 255, 0, 125), thickness=2)


    scriptOp.copyNumpyArray(img)

    return



