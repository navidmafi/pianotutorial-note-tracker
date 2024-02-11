import numpy as np
from sklearn.cluster import DBSCAN
from scipy.interpolate import splprep, splev
import typing
from typing import Tuple, List, Sequence, TypeVar, Annotated, Literal, Union
import cv2
from numpy import uint8

import pickle
import mido
from mido import MidiFile, MidiTrack, Message,MetaMessage

import cv2.typing as cvt
import numpy.typing as npt


DType = TypeVar("DType", bound=np.generic)
TRegionBox = Tuple[int,int,int,int]
TPoint = Tuple[int,int]
# TContourAnnotated = Annotated[npt.NDArray[DType], TPoint]
# TContour = List[TPoint]
TContour = cvt.MatLike
TContours = Sequence[TContour]
# TImageData = npt.NDArray[uint8]


TImage = cvt.MatLike


def flatmap_contour_points(contours: TContours) -> npt.NDArray:
    contour_coordinates = []
    for contour in contours:
        for point in contour:
            contour_coordinates.append(point[0])

    return np.array(contour_coordinates)



def get_largest_bbox(bboxes: List[TRegionBox]) -> TRegionBox:
    largest_bbox = None
    largest_area = 0

    for box in bboxes:
        x, y, w, h = box
        area = w * h

        if area > largest_area:
            largest_bbox = box
            largest_area = area
    
    if (largest_bbox) == None:
        raise Exception("bbox list empty thus no largest bbox was found")
    
    return largest_bbox


def getbboxes_dbscan(contour_coordinates: npt.NDArray) -> List[TRegionBox]:
    dbscan = DBSCAN(eps=60, min_samples=20)  # Adjust the parameters as needed
    cluster_labels = dbscan.fit_predict(contour_coordinates)

    bounding_boxes = []
    unique_labels = np.unique(cluster_labels)
    for label in unique_labels:
        if label == -1:  # ignore noise
            continue
        cluster_points : TContour = contour_coordinates[cluster_labels == label]
        bounding_boxes.append(contour_to_bounding_rect(cluster_points))

    return bounding_boxes
   

def filter_contours(
    contours: TContours, min_area=100, max_area=np.inf
):
    filtered_contours = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if min_area <= area <= max_area:
            filtered_contours.append(contour)
    return filtered_contours


def write_stroked_text(image, text, cx, cy):
    font_face = cv2.FONT_HERSHEY_PLAIN
    font_scale = 1
    color = (0,)
    thickness = 2
    stroke_color = (255,255,255)
    stroke_thickness = 2

    
    cv2.putText(image, text, (cx, cy), font_face, font_scale, stroke_color, thickness + stroke_thickness, cv2.LINE_AA)
    cv2.putText(image, text, (cx, cy), font_face, font_scale, color, thickness, cv2.LINE_AA)

    return image



class Image:
    def __init__(self, data: TImage):
        self.data = data

    def add_padding(self, padding_size):
        self.data = cv2.copyMakeBorder(self.data, padding_size, padding_size, padding_size, padding_size, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return self

    def add_border(self, border_size, border_color):
        self.data = cv2.copyMakeBorder(self.data, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=border_color)
        return self

    def display(self, data=None , window_name='Image'):
        if data is None:
            data = self.data
        cv2.imshow(window_name, data)
        while cv2.waitKey(0) != 27:
            pass
        cv2.destroyAllWindows()
        return self


    def threshold(self, threshold_value=127, max_value=255):
        _, self.data = cv2.threshold(self.data, threshold_value, max_value, cv2.THRESH_BINARY)
        return self


    def get_contours(self, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE) -> TContours:
        contours, _ = cv2.findContours(self.data, mode, method)
        return contours
    
    def get_canny_edges(self, threshold1=50, threshold2=150):
        self.data = cv2.Canny(self.data, threshold1, threshold2)
        return self

    def to_hsv(self):
        self.data = cv2.cvtColor(self.data, cv2.COLOR_BGR2HSV_FULL)
        return self
     
    def to_rgb(self):
        self.data = cv2.cvtColor(self.data, cv2.COLOR_BGR2RGB)
        return self
    
    def to_grayscale(self):
        self.data = cv2.cvtColor(self.data, cv2.COLOR_BGR2GRAY)
        return self
    


    def increase_contrast(self, alpha=1.0, beta=0):
        self.data = cv2.convertScaleAbs(self.data, alpha=alpha, beta=beta)
        return self

    def crop(self,x, y, w, h ):
        self.data = self.data[y:y+h,x:x+w]
        return self

    def crop_from(self, direction, border_size):
        height, width = self.data.shape[:2]

        if direction == "up":
            y_min = int(border_size * height)
            y_max = height
            x_min = 0
            x_max = width
        elif direction == "down":
            y_min = 0
            y_max = height - int(border_size * height)
            x_min = 0
            x_max = width
        elif direction == "left":
            y_min = 0
            y_max = height
            x_min = int(border_size * width)
            x_max = width
        elif direction == "right":
            y_min = 0
            y_max = height
            x_min = 0
            x_max = width - int(border_size * width)
        else:
            raise ValueError("Invalid direction. Valid options are 'up', 'down', 'left', or 'right'.")

        self.data = self.data[y_min:y_max, x_min:x_max]

        return self

    def crop_around(self, border_size : int):
        height, width = self.data.shape[:2]
        if border_size >= height or border_size >= width:
            raise ValueError("Border size is larger than image dimensions")
        
        cropped_data = self.data[border_size:height-border_size, border_size:width-border_size]
        self.data = cropped_data
        return self
    

    def deep_copy(self):
        return Image(self.data.copy())


def calculate_bounding_centroid(contour : TContour) -> tuple:
    x, y, w, h = contour_to_bounding_rect(contour)
    
    # Adjust contour points to account for offset
    bounding_contour = contour - np.array([x, y])
    
    # Calculate centroid using moments of the adjusted bounding contour
    M = cv2.moments(bounding_contour)
    
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"]) + x  # Add x offset back to centroid
        cy = int(M["m01"] / M["m00"]) + y  # Add y offset back to centroid
    else:
        cx, cy = 0, 0
    
    return cx, cy


def bounding_rect_to_contour(rect: TRegionBox) -> TContour:
    x, y, w, h = rect
    contour = np.array([
        [x, y],
        [x + w, y],
        [x + w, y + h],
        [x, y + h]
    ])
    return contour

def contour_to_bounding_rect(contour: TContour) -> TRegionBox:
    x, y, w, h = cv2.boundingRect(np.mat(contour))
    return (x,y,w,h)


def debug_show_contours(base_image : Image,contours : TContours, fill = True) -> None:
    img = base_image.deep_copy().to_rgb()
    
    for i, contour in enumerate(contours):
        cx, cy = calculate_bounding_centroid(contour)

        color = np.random.randint(0, 255, size=(3,))
        if (fill):
            cv2.drawContours(img.data, [np.mat(contour)], -1, color.tolist(), cv2.FILLED)
        cv2.drawContours(img.data, [np.mat(contour)], -1, (0, 0, 255), 1)
        write_stroked_text(img.data, str(i), cx - 10  , cy)

    img.display()

def debug_show_bboxes(base_image : Image, bboxes: List[TRegionBox]):
    imgCopy = base_image.deep_copy()
    for bbox in bboxes:
        x, y, w, h = bbox
        cv2.rectangle(imgCopy.data, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    imgCopy.display()

def debug_show_bboxes_dict(base_image: Image, key_rects : dict):
    img_copy = base_image.deep_copy()

    for key_number, rect in key_rects.items():
        x, y, w, h = rect
        cv2.rectangle(img_copy.data, (x, y), (x + w, y + h), (0, 255, 0), 2)

    img_copy.display()

def create_mask_from_contours(image_shape : tuple, contours : TContours):
    mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
    
    return mask


def reverse_mask(mask : TImage):
    # Create a white canvas with the same shape as the mask
    reversed_mask = np.ones_like(mask, dtype=np.uint8) * 255
    
    # Subtract the mask from the white canvas
    reversed_mask = cv2.bitwise_and(reversed_mask, np.array(255) - mask)
    
    return reversed_mask


def close_gaps(mask :TImage, kernel_size=5):
    # Define a kernel for morphological operations
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Perform dilation followed by erosion to close gaps
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return closed_mask

def sort_contours_horizontal(contours : List[TContour]):
    sorted_contours = sorted(contours, key=lambda c: contour_to_bounding_rect(c)[0])
    return sorted_contours


def merge_and_sort_contours(contours1 : TContours, contours2 : TContours):
    all_contours : List[TContour] = []
    all_contours += contours1
    all_contours += contours2
    
    sorted_contours = sort_contours_horizontal(all_contours)
    
    return sorted_contours




# def extract_key_bounding_rects(contours : TContours):
#     key_rects = {}  # Initialize an empty dictionary to store key bounding rects

#     for i, contour in enumerate(contours):
#         x, y, w, h = cv2.boundingRect(contour)
#         key_rects[i] = (x, y, w, h)  # Store the bounding rect in the dictionary with key number as the key

#     return key_rects


# #https://en.wikipedia.org/wiki/Shoelace_formula
# def triangle_area(p1, p2, p3):
#     if len(p1) < 2 or len(p2) < 2 or len(p3) < 2:
#         return 0  # Return 0 if any of the points does not have both x and y coordinates
#     return abs((p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1])) / 2)


# def simplify_contours_to_vertices(contours : TContours, num_vertices):
#     simplified_contours = []
#     for contour in contours:
#         areas = np.zeros(len(contour))
#         for i in range(1, len(contour) - 1):
#             areas[i] = triangle_area(contour[i - 1], contour[i], contour[i + 1])

#         sorted_indices = np.argsort(areas)[::-1]

#         simplified_contour = np.array(contour)[sorted_indices[:num_vertices]]
#         simplified_contours.append(simplified_contour)

#     return simplified_contours

def simplify_contours_douglas_peucker(contours : TContours, epsilon = 5):
    simplified_contours = [cv2.approxPolyDP(contour, epsilon, True) for contour in contours]
    return simplified_contours



def contours_to_list(contours : TContours):
    contour_list: List[TContour]= []
    for contour in contours:
        points = contour.squeeze().tolist()  
        contour_list.append(points)
    return contour_list

def get_key_color(image: Image, contour : TContour):
    mask = np.zeros(image.data.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, [255], cv2.FILLED)
    # Image(mask).to_rgb().display()
    mean = cv2.mean(image.data, mask=mask)
    return mean


def bottom_half_brect(contours: List[TContour], offset=10) -> List[TRegionBox]:
    bottom_half_contours : List[TRegionBox] = []
    for contour in contours:
        x, y, w, h = contour_to_bounding_rect(contour)
        bottom_half_contours.append((x, y + h // 2 + offset, w, h // 2 - offset))
    return bottom_half_contours

def scale_bounding_rect(rect: TRegionBox, scale_amount: float, direction='center') -> TRegionBox:
    x, y, width, height = rect

    center_x = x + width / 2
    center_y = y + height / 2

    # Scale the dimensions
    scaled_width = int(width * scale_amount)
    scaled_height = int(height * scale_amount)

    if direction == 'center':
        # Calculate the new top-left corner coordinates
        new_x = int(center_x - scaled_width / 2)
        new_y = int(center_y - scaled_height / 2)
    elif direction == 'bottom':
        new_x = x
        new_y = y
    elif direction == 'top':
        new_x = int(center_x - scaled_width / 2)
        new_y = int(y + height - scaled_height)
    elif direction == 'left':
        new_x = x
        new_y = int(center_y - scaled_height / 2)
    elif direction == 'right':
        new_x = int(x + width - scaled_width)
        new_y = int(center_y - scaled_height / 2)
    else:
        raise ValueError("Invalid direction. Choose from 'center', 'bottom', 'top', 'left', or 'right'.")

    return new_x, new_y, scaled_width, scaled_height



def crop_brect(brect: TRegionBox, from_direction: str , size : int):
    x, y, width, height = brect
    new_x = x
    new_y = y

    if from_direction == 'top':
        new_y = y
    elif from_direction == 'bottom':
        new_y = max(y + height - size, y)
    elif from_direction == 'left':
        new_x = x
    elif from_direction == 'right':
        new_x = max(x + width - size, x)
    else:
        raise ValueError("Invalid 'from_direction'. Choose from 'top', 'bottom', 'left', or 'right'.")

    if from_direction == 'top' or from_direction == 'bottom':
        new_x = max(x, x + (width - size) // 2)
    else:
        new_y = max(y, y + (height - size) // 2)

    new_width = min(width, size)
    new_height = min(height, size)

    return new_x, new_y, new_width, new_height




def adjust_gamma(image: TImage, gamma=1.0) -> TImage:
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def get_keys_colors(image: Image, black_key_contours: List[TContour], white_key_contours: List[TContour]) -> list:
    brects: List[TRegionBox] = [contour_to_bounding_rect(cnt) for cnt in key_contours]
    bh_key_contours = [bounding_rect_to_contour(crop_brect(crop_brect(scale_bounding_rect(brect,0.8),"bottom",50),"bottom", 45)) for brect in brects]
    colors = []


    # debug_show_bboxes(image,bh_regions)
    # debug_show_contours(image,bh_key_contours)
    src = image.deep_copy().to_grayscale()
    gamma = 0.3
    gamma_corrected = adjust_gamma(src.data,gamma)
    clahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(9, 9))
    clahe_image = clahe.apply(gamma_corrected)
    # Image(clahe_image).display()
    debug_show_contours(Image(gamma_corrected),bh_key_contours,False)

    debug_show_contours(Image(clahe_image),bh_key_contours,False)

    # Image(thresholded_image).display()
    for contour in bh_key_contours:
        average_color = get_key_color(Image(clahe_image), np.mat(contour))
        # print(average_color)
        colors.append(average_color)
    return colors



def detect_key_presses(curr_colors: List[tuple], base_colors: List[tuple], threshold: int = 90) -> list:
    key_presses = []

    for curr_color, base_color in zip(curr_colors, base_colors):
        distance = np.linalg.norm(np.array(curr_color) - np.array(base_color))
        
        if int(distance) > threshold:
            key_presses.append(True)  # pressed
        else:
            key_presses.append(False)  # not pressed

    return key_presses


def process_video(cap: cv2.VideoCapture, action : typing.Callable, frame_interval: int = 1, start = 0 , end = None) -> None:
    if end is None:
        end = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    frame_count = start
    while True:
        ret, frame = cap.read()
        if not ret:
            break 
        if (frame_count % frame_interval == 0):
            action(frame_count,frame) 
        frame_count += 1
        if frame_count >= end:
            break 




def convert_bool_to_midi(frames_per_second: int, bpm : int, notes_state_frame_map: dict[int, list[bool]], output_file: str) -> None:
    # tempo = mido.bpm2tempo(beats_per_minute)
    mspb = mido.bpm2tempo(bpm) #tempo
    tpb = 480
    midi_file = mido.MidiFile(ticks_per_beat=tpb)
    track = mido.MidiTrack()
    tempo_meta = MetaMessage('set_tempo', tempo=mspb)
    track.append(tempo_meta)

    current_notes = set()
    ticks_since_last_msg = 0
    for frame, note_states in notes_state_frame_map.items():
        # print(frame)
        for note, state in enumerate(note_states):
            # print(note, state,ticks_since_last_msg)
            if state:  # Note is on
                if note not in current_notes and state:
                    track.append(Message('note_on', note=note+21, velocity=127, time=ticks_since_last_msg))
                    current_notes.add(note)
                    ticks_since_last_msg = 0
            else:  # Note is off
                if note in current_notes and not state:
                    track.append(Message('note_off', note=note+21, velocity=127, time=ticks_since_last_msg))
                    current_notes.remove(note)
                    ticks_since_last_msg = 0
            
        ticks_since_last_msg += int(1000/frames_per_second)

                
        
        
                    

    midi_file.tracks.append(track)

    # Save MIDI file
    midi_file.save(output_file)



def get_nth_frame(cap: cv2.VideoCapture, n) -> TImage:
    cap.set(cv2.CAP_PROP_POS_FRAMES, n - 1) 
    _, frame = cap.read()
    return frame
    



def get_roi_box(base_image: Image) -> TRegionBox:
    h,w = base_image.data.shape[:2]
    roi_ready_image = base_image.deep_copy().to_grayscale().threshold(140,250).get_canny_edges()
    roi_contours = roi_ready_image.get_contours()
    # debug_show_contours(roi_ready_image, roi_contours)
    # largest_contour = max(roi_contours, key=cv2.contourArea)
    # debug_show_contours(base_image, [largest_contour])
    contour_points = flatmap_contour_points(roi_contours)
    # exit()
    bboxes = getbboxes_dbscan(contour_points)
    # debug_show_bboxes(base_image,bboxes)
    target_bbox_roi = get_largest_bbox(bboxes)
    # debug_show_bboxes(base_image,[target_bbox_roi])
    # exit()
    return target_bbox_roi



def get_key_contours(roi_image: Image) -> Tuple[List[TContour],List[TContour]]: #white, black
    contour_img = roi_image.deep_copy().to_grayscale().threshold(110,255)
    # debug_show_bboxes(complexImage.deep_copy().to_rgb(),[target_bbox_roi])
    white_keys_contours = filter_contours(contour_img.get_contours(),1000)

    # debug_show_contours(roi_image,white_keys_contours)

    contoured_mask = create_mask_from_contours(roi_image.data.shape,white_keys_contours)
    not_contoured_mask = Image(reverse_mask(close_gaps(contoured_mask, 10)))

    black_keys_contours = filter_contours(not_contoured_mask.deep_copy().to_grayscale().get_contours(),200)
    # debug_show_contours(roi_image,black_keys_contours)

    # all_keys_contours = filter_contours(merge_and_sort_contours(white_keys_contours, black_keys_contours),500)
    # simplified_contours = filter_contours(simplify_contours_douglas_peucker(all_keys_contours,2))

    debug_show_contours(roi_image,simplified_contours)
    # bounding_rects = extract_key_bounding_rects(simplified_contours)

    key_contours = contours_to_list(simplified_contours)
    return key_contours

    


# AYAYA=7598

# # read_video("src.mp4",action,1,AYAYA,10004)

# file_path = 'key_value_data.pkl'

# # with open(file_path, 'wb') as f:
# #     pickle.dump(playback_frames, f)

# # frames_to_midi(60,124,playback_frames)
    
# with open(file_path, 'rb') as f:
#     loaded_kv_data = pickle.load(f)

# # Now loaded_kv_data contains the same key-value pairs as kv_data
# print("Loaded key-value pairs")
# generated_midi = convert_bool_to_midi(60, 124,loaded_kv_data,'output.mid')






def main():
    input_vid = "src2.mkv"
    start_time_sec = 450/60
    end_time_sec = start_time_sec + 4
    FORCE_CACHE = False

    video_cap = cv2.VideoCapture(input_vid)

    if not video_cap.isOpened():
        print("Error: Couldn't open the video file.")
        return
    

    video_fps = video_cap.get(cv2.CAP_PROP_FPS)
    print(video_fps)
    # base_frame_index = int(start_time_sec * video_fps)
    base_frame_index = 153

    base_frame_image = Image(get_nth_frame(video_cap,base_frame_index)).crop_from("up",4/8)

    roi_box = get_roi_box(base_frame_image)
    roi_image = base_frame_image.deep_copy().crop(*roi_box)
    key_contours = get_key_contours(roi_image)


    base_avg_colors = get_keys_colors(roi_image,key_contours)
    playback_data={}

    def action(frameNumber: int,frame:TImage):
        frame_img = Image(frame).crop_from("up",4/8)
        frame_img.crop(*roi_box)
        frame_colors = get_keys_colors(frame_img,key_contours)
        
        keypresses = detect_key_presses(frame_colors,base_avg_colors)
        print(f'processed frame {frameNumber}')
        print(f'keys pressed : {[i for i,kp in enumerate(keypresses) if kp == True]}')
        playback_data[frameNumber] = keypresses

    process_video(video_cap,action,1, int(start_time_sec * video_fps), int(end_time_sec * video_fps))

    convert_bool_to_midi(int(video_fps),90,playback_data,"moo.mid")



    


  
main()

