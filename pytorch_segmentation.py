# me - this DAT
# scriptOp - the OP which is cooking

import time
import numpy as np
from colorsys import hls_to_rgb
from ultralytics import YOLO
import torch 
import cv2
from ultralytics import YOLO







def bitMaskSegmentation(resultsZero):

    # Sum all masks and add a channel dimension
    sum_masks = torch.sum(resultsZero.masks.data, dim=0).unsqueeze(0)

    # Transpose the dimensions and convert to HSV
    mask_raw = sum_masks.cpu().numpy().transpose(1, 2, 0)
    hsv = cv2.cvtColor(mask_raw, cv2.COLOR_GRAY2BGR)

    # Create and invert the mask
    mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([0, 0, 1]))
    mask = cv2.bitwise_not(mask)

    # Ensure the data type of the original image is uint8 (CV_8U)
    orig_img_uint8 = resultsZero.orig_img.astype(np.uint8)

    # Resize the mask to the same size as the original image
    mask = cv2.resize(mask, (orig_img_uint8.shape[1], orig_img_uint8.shape[0]))

    # Apply the mask to the original image
    #masked = cv2.bitwise_and(orig_img_uint8, orig_img_uint8, mask=mask)

    return mask









Modelpath = str(op('script2').par.Onnxmodel)
#ort_session = onnxruntime.InferenceSession(Modelpath)
print("INIT")

#print(torch.cuda.get_device_name(0))
# Load the YOLOv8 model
model = YOLO(Modelpath) # fastest model
print(torch.cuda.get_device_name(0))

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


def onCook(scriptOp):
    
    #img = scriptOp.inputs[0].numpyArray(delayed=True)
    img = scriptOp.inputs[0].numpyArray()
    img_scaled = (img[:, :, :3] * 255).astype(np.uint8)
    #img_scaled = np.transpose(img_scaled, (1, 0, 2))
   

    t1 = time.time() #time
    results = model.predict(source=img_scaled, classes=0, conf=0.3, verbose=False)
                
    if results[0].masks is not None:
        frame = bitMaskSegmentation(results[0])
    else:
        # If no masks are found, assign a black frame
        frame = np.zeros_like(frame)  # Creates a black frame with the same shape as the original frame
    

    imgRGBA = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGBA)


    t2 = time.time()  # time
    scriptOp.store("time", (t2 - t1) * 1000)  # time
 
    scriptOp.copyNumpyArray(imgRGBA)

    return
