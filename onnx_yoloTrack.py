
import math
import time
import cv2
import numpy as np
import onnxruntime

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

class MultiObjectTracker:

    def __init__(self, track_persistance: int = 1, 
                minimum_track_length: int = 1,
                iou_lower_threshold: float = 0.04,
                interpolate_tracks: bool = False,
                cross_class_tracking: bool = True):

        assert track_persistance >= 0
        assert 0 <= iou_lower_threshold <= 1
        assert minimum_track_length > 0

        self.track_persistance = track_persistance
        self.iou_lower_threshold = iou_lower_threshold
        self.minimum_track_length = minimum_track_length
        self.interpolate_tracks = interpolate_tracks
        self.cross_class_tracking = cross_class_tracking

        self.active_tracks = []
        self.finished_tracks = []

        self.finished_tracking = False
        self.displayed_finished_tracking_warning = False

        self.time_step = 0

    def step(self,list_of_new_boxes: list = []):
        ''' Call this method to add more bounding boxes to the tracker'''

        self.time_step += 1 #increment time step

        #build hungarian matrix of bbox IOU's
        hungarian_matrix = np.zeros((len(self.active_tracks), len(list_of_new_boxes)))

        if len(self.active_tracks) > 0 and len(list_of_new_boxes) > 0:
            active_boxes = np.concatenate([track.get_next_predicted_box()[np.newaxis,:]
                            for track in self.active_tracks], axis = 0)

            new_boxes = np.concatenate([ b['box'][np.newaxis,:] for b in list_of_new_boxes], 
                                        axis = 0)
            hungarian_matrix = IOU(new_boxes,active_boxes)

            #print(hungarian_matrix)

            #zero out IOU's less than IOU min threshold to prevent assigment
            hungarian_matrix[hungarian_matrix < self.iou_lower_threshold] = 0

            #zero out IOU's where the object_class variables don't match
            if not self.cross_class_tracking:
                for i,new_box in enumerate(list_of_new_boxes):
                    if "object_class" in new_box:
                        for j,active_track in enumerate(self.active_tracks):
                            active_box = active_track.boxes[0] #shouldn't matter which box in the track
                            if "object_class" in active_box and \
                                are_coco_classes_different(new_box['object_class'], active_box['object_class']):
                                hungarian_matrix[i,j] = 0

            #print(hungarian_matrix)

            #compute optimal box matches with Hungarian algorithm
            row_ind, col_ind = linear_sum_assignment(hungarian_matrix, maximize = True)

            #assign new boxes to active tracks, boxes not assigned this way become new tracks
            for i,box in enumerate(list_of_new_boxes):
                if i in row_ind:
                    #if the new box has matching active track, add it to that track
                    r = (row_ind==i).argmax(axis=0)
                    c = col_ind[r]
                    if hungarian_matrix[r,c] > 0:
                        self.active_tracks[c].add_box(box, self.time_step)
                    else:
                        #new box has no matching active track, create a new track for it
                        self.active_tracks.append(Track(initial_box = box, initial_time_step = self.time_step))
                else:
                    #new box has no matching active track, create a new track for it (same as row above)
                    self.active_tracks.append(Track(initial_box = box, initial_time_step = self.time_step))

        else:
            for box in list_of_new_boxes:
                self.active_tracks.append(Track(initial_box = box, initial_time_step = self.time_step))

        #active tracks with age >= track_persistance are set to be finished tracks
        newly_finished_track_ind = []
        for i, trk in enumerate(self.active_tracks):
            if trk.timestamps[-1] + self.track_persistance < self.time_step:
                self.finished_tracks.append(trk)
                newly_finished_track_ind.append(i)

        self.active_tracks = [element for i,element in enumerate(self.active_tracks) if i not in newly_finished_track_ind]

    def finish_tracking(self):
        ''' Call this method when you are done adding new boxes.

        Finish all active tracks, do interpolation if selected, prune tracks shorter than 
        minimum_track_length parameter '''
        self.finished_tracks.extend(self.active_tracks)
        self.active_tracks = []

        if self.interpolate_tracks:
            for i in range(len(self.finished_tracks)):
                self.finished_tracks[i].interpolate()

        #prune tracks with length less than minimum_track_length
        self.finished_tracks = [trk for trk in self.finished_tracks if len(trk) >= self.minimum_track_length]

        #sort tracks by track start time
        self.finished_tracks = sorted(self.finished_tracks, key = lambda x: x.timestamps[0])

        self.finished_tracking = True

    def export_pandas_dataframe(self, additional_cols = 'auto'):
        '''Converts multi-object tracker internal state to Pandas dataframe with cols:
        Time, TrackID, X1, Y1, X2, Y2

        Arguments:
        additional_cols {list or str} -- List of additional attributes to grab from the Box class. 
                                For each attribute, a column will be added to dataframe and
                                the function will attempt to grab that value from each Box object
                                added to the dataframe. If the attribute is not present in a
                                Box object, it will add NaN or None.

                                If this variable is set to the string "auto", it will populate this
                                variable with all extra params in all Box added to the tracker. 

        Returns:
        Pandas DataFrame with specified rows'''

        if not self.finished_tracking and not self.displayed_finished_tracking_warning:
            print('[WARNING] -- Exporting Pandas DataFrame without calling finish_tracking(). Active tracks will not be exported.')
            self.displayed_finished_tracking_warning = True

        if isinstance(additional_cols,str) and additional_cols in ['Auto','auto']:
            additional_cols = set()

            for trk in self.finished_tracks:
                for box in trk.boxes:
                    additional_cols = additional_cols.union(set(box.keys()))

            additional_cols.discard("box")
            additional_cols = list(additional_cols)

        df_list = []

        for time in range(self.time_step + 1):
            for track_id, trk in enumerate(self.finished_tracks):
                if time in trk.timestamps:
                    box = trk.boxes[trk.timestamps.index(time)]
                    row = {'Time': time,
                            'TrackID': track_id,
                            'X1': box['box'][0],
                            'Y1': box['box'][1],
                            'X2': box['box'][2],
                            'Y2': box['box'][3],
                            }
                    for col in additional_cols:
                        row[col] = box[col] if col in box.keys() else None

                    df_list.append(row)

        return pd.DataFrame(df_list)

    def print_internal_state(self):
        ''' For debugging purposes'''

        print('###########################################')
        print(f'Time Step: {self.time_step}')
        print('---------------Active Tracks---------------')
        for i,trk in enumerate(self.active_tracks):
            print(f'Track {i}: { [list(b["box"]) for b in trk.boxes] }')

        print('--------------Finished Tracks--------------')
        for i,trk in enumerate(self.finished_tracks):
            print(f'Track {i}: {[list(b["box"]) for b in trk.boxes]}')
        print('###########################################')


class Track:

    def __init__(self, initial_box, initial_time_step):
        self.boxes = [initial_box]
        self.timestamps = [initial_time_step]

    def get_next_predicted_box(self):
        #maybe add a kalman filter here or something

        return self.boxes[-1]['box']

    def add_box(self, box, time_step):
        self.boxes.append(box)
        self.timestamps.append(time_step)

    def interpolate(self):
        ''' TODO: Implement this function'''
        interpolated_boxes = []

        for i in range(len(self.boxes) - 1):
            interpolated_boxes.append(self.boxes[i])

            delta = self.timestamps[i+1] - self.timestamps[i]
            if delta == 1:
                continue #no need to interpolate sequential boxes

            #not interpolating confidence or other numerical parameters
            new_boxes = [{"box": ((delta - j) * self.boxes[i]['box'] + j * self.boxes[i+1]['box'])/delta}
                            for j in range(1, delta)]

            interpolated_boxes.extend(new_boxes)

        interpolated_boxes.append(self.boxes[-1])
        self.boxes = interpolated_boxes
        self.timestamps = list(range(self.timestamps[0], self.timestamps[-1] + 1))

        assert len(self.boxes) == len(self.timestamps), f'length of boxes ({len(self.boxes)} != length timestamps ({len(self.timestamps)})'


    def __len__(self):
        return self.timestamps[-1] - self.timestamps[0] + 1

def IOU(bboxes1, bboxes2, isPixelCoord = 1):
    #vectorized IOU numpy code from:
    #https://medium.com/@venuktan/vectorized-intersection-over-union-iou-in-numpy-and-tensor-flow-4fa16231b63d

    #input N x 4 numpy arrays
    x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    interArea = np.maximum((xB - xA + isPixelCoord), 0) * np.maximum((yB - yA + isPixelCoord), 0)
    boxAArea = (x12 - x11 + isPixelCoord) * (y12 - y11 + isPixelCoord)
    boxBArea = (x22 - x21 + isPixelCoord) * (y22 - y21 + isPixelCoord)
    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
    return iou

def are_coco_classes_different(c1, c2):
    ''' Sets which classes are "equivalent" for the tracker. Most classes are not equal
    but objects like pickup trucks can be detected as both car and truck. Or sometimes
    busses can be detected as both truck and bus. This function is a quick and dirty
    way to stop the track from breaking.
    '''
    return c1 != c2 and ( {c1,c2} not in [{'car','truck'}, {'bus','truck'}])




class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# Create a list of colors for each class where each color is a tuple of 3 integer values
rng = np.random.default_rng(3)
colors = rng.uniform(0, 255, size=(len(class_names), 3))


def nms(boxes, scores, iou_threshold):
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # Compute IoU of the picked box with the rest
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious < iou_threshold)[0]

        # print(keep_indices.shape, sorted_indices.shape)
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes

def multiclass_nms(boxes, scores, class_ids, iou_threshold):

    unique_class_ids = np.unique(class_ids)

    keep_boxes = []
    for class_id in unique_class_ids:
        class_indices = np.where(class_ids == class_id)[0]
        class_boxes = boxes[class_indices,:]
        class_scores = scores[class_indices]

        class_keep_boxes = nms(class_boxes, class_scores, iou_threshold)
        keep_boxes.extend(class_indices[class_keep_boxes])

    return keep_boxes


def compute_iou(box, boxes):
    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area

    return iou


def xywh2xyxy(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def draw_detections(image_shape, boxes, scores, class_ids, mask_alpha=0.3, mask_maps=None):

    img_height, img_width = image_shape[:2]

    size = min([img_height, img_width]) * 0.0006
    text_thickness = int(min([img_height, img_width]) * 0.001)
    
    black_image = np.zeros((img_height, img_width, 3), dtype=np.float32)

    mask_img = draw_masks(black_image, boxes, class_ids, mask_alpha, mask_maps)
    # for class_id in class_ids:
    #     label = class_names[class_id]
    #     print(label)

    #if len(mask_maps):
    #    return mask_img 
    #Draw bounding boxes and labels of detections

    split_color_boxes = np.zeros_like(mask_img)

    coverage_threshold = 0.3
    for box, score, class_id in zip(boxes, scores, class_ids):
        color = colors[class_id].astype(int)/255  # Assuming colors is a NumPy array


        x1, y1, x2, y2 = box.astype(int)
        # Calculate the area of the bounding box
        box_area = (x2 - x1) * (y2 - y1)

        # Calculate the percentage of the image covered by the bounding box
        coverage_percentage = box_area / (img_height * img_width)
        if  bool(op('script2').par.Drawboxes):
            # Exclude boxes that cover more than 30%
            if coverage_percentage <= coverage_threshold:
                if bool(op('script2').par.Splitcolors): 
                    color = (0,255,0)
                    # Draw rectangle
                    cv2.rectangle(split_color_boxes, (x1, y1), (x2, y2), color, 2)
                else:
                    cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, 2)

        # label = class_names[class_id]
        # caption = f'{label} {int(score * 100)}%'
        # (tw, th), _ = cv2.getTextSize(text=caption, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        #                               fontScale=size, thickness=text_thickness)
        # th = int(th * 1.2)

        # cv2.rectangle(mask_img, (x1, y1),
        #               (x1 + tw, y1 - th), color, -1)

        # cv2.putText(mask_img, caption, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)
        
       
    if bool(op('script2').par.Splitcolors):
        mask_img = cv2.add(mask_img,split_color_boxes)
        


    return mask_img



def draw_masks(image, boxes, class_ids, mask_alpha=0.3, mask_maps=None, draw_contours=False):
    mask_img = np.zeros_like(image)
    countours_img = np.zeros_like(image)
    img_height,img_width = image.shape[:2]
    coverage_threshold = 0.3

    # Draw bounding boxes and labels of detections
    for i, (box, class_id) in enumerate(zip(boxes, class_ids)):
        color = colors[class_id]

        x1, y1, x2, y2 = box.astype(int)
        # Calculate the area of the bounding box
        box_area = (x2 - x1) * (y2 - y1)

        # Calculate the percentage of the image covered by the bounding box
        coverage_percentage = box_area / (img_height * img_width)

        # Exclude boxes that cover more than 30%
        if coverage_percentage <= coverage_threshold:
            if mask_maps is not None:
                crop_mask = mask_maps[i][y1:y2, x1:x2, np.newaxis]
                crop_mask_img = mask_img[y1:y2, x1:x2]
                color = colors[class_id].astype(int)/255  # Assuming colors is a NumPy array
                if bool(op('script2').par.Splitcolors): 
                    color = (255,0,0)

                crop_mask_img = crop_mask_img * (1 - crop_mask) + crop_mask * color
                mask_img[y1:y2, x1:x2] = crop_mask_img

            # contours, _ = cv2.findContours(crop_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # # Offset the contours by x1 and y1
            # offset_contours = [contour + (x1, y1) for contour in contours]
            # cv2.drawContours(countours_img, offset_contours, -1, color, 2)

    

    return mask_img



class YOLOv8:

    def __init__(self, path, conf_thres=0.7, iou_thres=0.5):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres

        # Initialize model
        self.initialize_model(path)

    def __call__(self, image):
        return self.detect_objects(image)

    def initialize_model(self, path):
        self.session = onnxruntime.InferenceSession(path,
                                                    providers=onnxruntime.get_available_providers())
        # Get model info
        self.get_input_details()
        self.get_output_details()

    def direct_call(self, input_tensor,image_shape):
        self.img_height,  self.img_width, nchan = image_shape

        return self.direct_detect_objects(input_tensor)

    def direct_detect_objects(self, input_tensor):
        # Perform inference on the image
        outputs = self.inference(input_tensor)

        self.boxes, self.scores, self.class_ids = self.process_output(outputs)

        return self.boxes, self.scores, self.class_ids, self.mask_maps

    def detect_objects(self, image):
        input_tensor = self.prepare_input(image)

        # Perform inference on the image
        outputs = self.inference(input_tensor)

        self.boxes, self.scores, self.class_ids = self.process_output(outputs)

        return self.boxes, self.scores, self.class_ids

    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor


    def inference(self, input_tensor):
        start = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})

        # print(f"Inference time: {(time.perf_counter() - start)*1000:.2f} ms")
        return outputs

    def process_output(self, output):
        predictions = np.squeeze(output[0]).T

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return [], [], []

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(predictions)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        # indices = nms(boxes, scores, self.iou_threshold)
        indices = multiclass_nms(boxes, scores, class_ids, self.iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices]

    def extract_boxes(self, predictions):
        # Extract boxes from predictions
        boxes = predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes)

        # Convert boxes to xyxy format
        boxes = xywh2xyxy(boxes)

        return boxes

    def rescale_boxes(self, boxes):

        # Rescale boxes to original image dimensions
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return boxes

    def draw_detections(self, image, draw_scores=True, mask_alpha=0.4):

        return draw_detections(image, self.boxes, self.scores,
                               self.class_ids, mask_alpha)

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

Modelpath = str(op('script2').par.Onnxmodel)

yolov8_detector = YOLOv8(Modelpath, conf_thres=0.2, iou_thres=0.3)

# font_path = str(op('script2').par.Font)

# font_size = 20
# font = ImageFont.truetype(font_path, font_size)

print("   ")
print("   ")
print( "Using device",onnxruntime.get_device()  )

mot = MultiObjectTracker(track_persistance = 2, 
                         minimum_track_length = 2, 
                         iou_lower_threshold = 0.04, 
                         interpolate_tracks = True)


def onCook(scriptOp):
    print("   ")
    print("   ")
    print("cook")

    img = scriptOp.inputs[0].numpyArray()
    img_copy_CV = img[:, :, :3]
    input = img_copy_CV.transpose(2, 0, 1).reshape(1, 3, 640, 640).astype("float32")

    
    conf = float(op('script2').par.Conf)
    yolov8_detector.conf_threshold = conf

    # Run YOLOv8 model
    resX = int(op('info')["resx"])
    resY = int(op('info')["resy"])
    orig_img_shape = (resY,resX,4)
    start_time = time.time()


    boxes, scores, class_ids, masks = yolov8_detector.draw_detections(input,orig_img_shape) # 30 millisecond

    formatted_boxes = []
    for box, score, class_id, mask in zip(boxes, scores, class_ids, masks):
        confidence = float(score)
        object_class = class_id
        formatted_box = {
            "box": np.array([box[0], box[1], box[2], box[3]]),  # Convert to (x_min, y_min, x_max, y_max) format
            "confidence": confidence,
            "object_class": object_class
        }
        formatted_boxes.append(formatted_box)

    # Use the formatted_boxes in your following steps
    mot.step(formatted_boxes)
    #mot.print_internal_state()

    combined_img = yoloseg.draw_masks(orig_img_shape, mask_alpha=0.5) # 23 millisecond
    split_color_tracking = np.zeros_like(combined_img)
        
    for track in mot.active_tracks:
        track_boxes = track.boxes

    
        for i, track in enumerate(track_boxes):

            
            box = track["box"]
            confidence = track.get("confidence", 0)
            object_class = track.get("object_class", "Unknown")
            # Convert box coordinates to integers
            box = box.astype(int)
            centroid = [(box[0] + box[2]) // 2, (box[1] + box[3]) // 2]

            # Check if it's the last element
            if  bool(op('script2').par.Drawlabel): 
                if i == len(track_boxes) - 1:
                    # Draw label and confidence text on the image
                    label = f"{class_names[object_class]}: {confidence:.2f}"
                    cv2.putText(combined_img, label, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            radius = min(30,int((i + 1) * 0.5))  # Gradually increase the size of the circles
            # Draw circle on the image
            color = colors[object_class].astype(int)/255  # Assuming colors is a NumPy array

            if bool(op('script2').par.Splitcolors): 
                color = (0,0,255)
                cv2.circle(split_color_tracking, centroid, radius, color, -1)
            else:
                cv2.circle(combined_img, centroid, radius, color, -1)

    if bool(op('script2').par.Splitcolors):
        combined_img = cv2.add(combined_img,split_color_tracking)

    scriptOp.copyNumpyArray(combined_img)

    return

def onSetupParameters(scriptOp):
	"""Auto-generated by Component Editor"""
	# manual changes to anything other than parJSON will be	# destroyed by Comp Editor unless doc string above is	# changed

	TDJSON = op.TDModules.mod.TDJSON
	parJSON = """
	{
		"Fast Neural Style": {
			"Font": {
				"name": "Font",
				"label": "font",
				"page": "Fast Neural Style",
				"style": "File",
				"default": "",
				"enable": true,
				"startSection": false,
				"readOnly": false,
				"enableExpr": null,
				"help": ""
			},
			"Drawlabel": {
				"name": "Drawlabel",
				"label": "DrawLabel",
				"page": "Fast Neural Style",
				"style": "Toggle",
				"default": false,
				"enable": true,
				"startSection": false,
				"readOnly": false,
				"enableExpr": null,
				"help": ""
			},
			"Splitcolors": {
				"name": "Splitcolors",
				"label": "SplitColors",
				"page": "Fast Neural Style",
				"style": "Toggle",
				"default": false,
				"enable": true,
				"startSection": false,
				"readOnly": false,
				"enableExpr": null,
				"help": ""
			},
			"Drawboxes": {
				"name": "Drawboxes",
				"label": "DrawBoxes",
				"page": "Fast Neural Style",
				"style": "Toggle",
				"default": false,
				"enable": true,
				"startSection": false,
				"readOnly": false,
				"enableExpr": null,
				"help": ""
			},
			"Onnxmodel": {
				"name": "Onnxmodel",
				"label": "ONNX Model",
				"page": "Fast Neural Style",
				"style": "File",
				"default": "",
				"enable": true,
				"startSection": false,
				"readOnly": false,
				"enableExpr": null,
				"help": ""
			},
			"Conf": {
				"name": "Conf",
				"label": "Conf",
				"page": "Fast Neural Style",
				"style": "Float",
				"size": 1,
				"default": 0.0,
				"enable": true,
				"startSection": false,
				"readOnly": false,
				"enableExpr": null,
				"help": "",
				"min": 0.0,
				"max": 1.0,
				"normMin": 0.0,
				"normMax": 1.0,
				"clampMin": false,
				"clampMax": false
			}
		}
	}
	"""
	parData = TDJSON.textToJSON(parJSON)
	TDJSON.addParametersFromJSONOp(scriptOp, parData, destroyOthers=True)