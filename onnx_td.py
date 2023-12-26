# me - this DAT
# scriptOp - the OP which is cooking

import time

import numpy as np
import onnxruntime as onnxruntime

op('script2').store("device",onnxruntime.get_device())

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
	#print(x.shape)
#OXXN
	t1 = time.time() #time

	ort_inputs = {ort_session.get_inputs()[0].name: x}
	ort_outs = ort_session.run(None, ort_inputs)
	result= ort_outs[0][0]

	t2 = time.time() #time
	scriptOp.store("time",(t2-t1)*1000) #time
#Postprocessing steps
	
	result = np.clip(result, 0, 255)
	result = result.transpose(1,2,0).astype("uint8")
	alpha = np.full((width,height), 255, dtype=np.uint8)
	result = np.dstack((result, alpha))
	result = np.ascontiguousarray(result)
	#print(result.shape)
	#print(result)
	scriptOp.copyNumpyArray(result)
	return
