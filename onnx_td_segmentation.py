# me - this DAT
# scriptOp - the OP which is cooking

import time

import numpy as np
import onnxruntime as onnxruntime
import cv2

import cv2
import numpy as np

def visualize_segmentation(ort_output):
    # Combine channels into a single mask
    combined_mask = np.sum(ort_output[0], axis=0)

    # Normalize the mask to the range [0, 255]
    normalized_mask = ((combined_mask - combined_mask.min()) / (combined_mask.max() - combined_mask.min()) * 255).astype(np.uint8)

    # Create a color map for visualization
    colormap = cv2.applyColorMap(normalized_mask, cv2.COLORMAP_JET)

    return colormap



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
    x = np.array(img[:, :, 0:3])#.astype('float32')
    x = np.transpose(x, [2, 0, 1])
    x = np.expand_dims(x, axis=0)

    t1 = time.time() #time
    ort_inputs = {ort_session.get_inputs()[0].name: x}
    ort_outs = ort_session.run(None, ort_inputs)
    # Visualize segmentation
    colormap = visualize_segmentation(ort_outs[1])
    print(colormap.shape)
    t2 = time.time()  # time
    scriptOp.store("time", (t2 - t1) * 1000)  # time
 
    scriptOp.copyNumpyArray(colormap)

    return
