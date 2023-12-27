# me - this DAT
# scriptOp - the OP which is cooking

import time

import numpy as np
import onnxruntime as onnxruntime
import cv2

def visualize_segmentation(ort_output):
    # Assuming ort_output is of shape (1, C, H, W)
    num_channels = ort_output.shape[1]

    # Combine channels into a single mask (you may need to adjust this based on your model's output structure)
    combined_mask = np.sum(ort_output[0], axis=0)

    # Normalize the mask
    normalized_mask = (combined_mask - np.min(combined_mask)) / (np.max(combined_mask) - np.min(combined_mask)) * 255

    # Convert to uint8
    normalized_mask = normalized_mask.astype(np.uint8)

    # Create a color map for visualization
    colormap = cv2.applyColorMap(normalized_mask, cv2.COLORMAP_JET)

    return colormap

#op('script2').store("device",onnxruntime.get_device())

Modelpath = str(op('script2').par.Onnxmodel)
ort_session = onnxruntime.InferenceSession(Modelpath)

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

#preprocessing steps
    #img = scriptOp.inputs[0].numpyArray(delayed=True)
    img = scriptOp.inputs[0].numpyArray()
    width, height = img.shape[0:2]
    x = np.array(img[:, :, 0:3])#.astype('float32')
    x = np.transpose(x, [2, 0, 1])
    x = np.expand_dims(x, axis=0)

    t1 = time.time() #time
    ort_inputs = {ort_session.get_inputs()[0].name: x}
    ort_outs = ort_session.run(None, ort_inputs)
    # Visualize segmentation
    colormap = visualize_segmentation(ort_outs[1])

    t2 = time.time()  # time
    scriptOp.store("time", (t2 - t1) * 1000)  # time
 
    scriptOp.copyNumpyArray(colormap)

    return
