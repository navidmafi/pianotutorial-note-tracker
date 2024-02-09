import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from scipy.interpolate import splprep, splev
import typing
import cv2 as cv
from numpy import uint8
from numpy.typing import NDArray
import cv2.typing
import pickle
import mido
from mido import MidiFile, MidiTrack, Message,MetaMessage

TImageData = NDArray[uint8]

def flatmap_contour_points(contours):
    contour_coordinates = []
    for contour in contours:
        for point in contour:
            contour_coordinates.append(point[0])

    contour_coordinates = np.array(contour_coordinates)

    return contour_coordinates



def get_largest_bbox(bboxes) -> tuple:
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


def getbboxes_dbscan(contour_coordinates):
    dbscan = DBSCAN(eps=80, min_samples=10)  # Adjust the parameters as needed
    cluster_labels = dbscan.fit_predict(contour_coordinates)

    bounding_boxes = []
    unique_labels = np.unique(cluster_labels)
    for label in unique_labels:
        if label == -1:  # ignore noise
            continue
        cluster_points = contour_coordinates[cluster_labels == label]
        x, y, w, h = cv.boundingRect(cluster_points)
        bounding_boxes.append((x, y, w, h))

    return bounding_boxes
   

def filterContours(
    contours, min_area=100, max_area=np.inf, min_aspect_ratio=7, max_aspect_ratio=10
):
    filtered_contours = []

    for contour in contours:
        area = cv.contourArea(contour)

        if min_area <= area <= max_area:
            # Calculate bounding box dimensions
            x, y, w, h = cv.boundingRect(contour)

            aspect_ratio = float(w) / h

            if min_aspect_ratio <= aspect_ratio <= max_aspect_ratio:
                filtered_contours.append(contour)
    return filtered_contours


def write_stroked_text(image, text, cx, cy):
    font_face = cv.FONT_HERSHEY_PLAIN
    font_scale = 1
    color = (0,)
    thickness = 2
    stroke_color = (255,255,255)
    stroke_thickness = 2

    
    cv.putText(image, text, (cx, cy), font_face, font_scale, stroke_color, thickness + stroke_thickness, cv.LINE_AA)
    cv.putText(image, text, (cx, cy), font_face, font_scale, color, thickness, cv.LINE_AA)

    return image



class Image:
    def __init__(self, data: cv2.typing.MatLike):
        self.data = data

    def add_padding(self, padding_size):
        self.data = cv.copyMakeBorder(self.data, padding_size, padding_size, padding_size, padding_size, cv.BORDER_CONSTANT, value=(0, 0, 0))
        return self

    def add_border(self, border_size, border_color):
        self.data = cv.copyMakeBorder(self.data, border_size, border_size, border_size, border_size, cv.BORDER_CONSTANT, value=border_color)
        return self

    def display(self, data=None , window_name='Image'):
        if data is None:
            data = self.data
        cv.imshow(window_name, data)
        while cv.waitKey(0) != 27:
            pass
        cv.destroyAllWindows()
        return self


    def threshold(self, threshold_value=127, max_value=255):
        _, self.data = cv.threshold(self.data, threshold_value, max_value, cv.THRESH_BINARY)
        return self


    def get_contours(self, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_SIMPLE) -> typing.Sequence[cv2.typing.MatLike]:
        contours, _ = cv.findContours(self.data, mode, method)
        return contours
    
    def get_canny_edges(self, threshold1=50, threshold2=150):
        self.data = cv.Canny(self.data, threshold1, threshold2)
        return self
    
    def to_rgb(self):
        self.data = cv.cvtColor(self.data, cv.COLOR_BGR2RGB)
        return self
    
    def to_grayscale(self):
        self.data = cv.cvtColor(self.data, cv.COLOR_BGR2GRAY)
        return self
    
    def crop(self,x, y, w, h ):
        self.data = self.data[y:y+h,x:x+w]
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


def calculate_bounding_centroid(contour : cv2.typing.MatLike) -> tuple:
    x, y, w, h = cv.boundingRect(contour)
    
    # Adjust contour points to account for offset
    bounding_contour = contour - np.array([x, y])
    
    # Calculate centroid using moments of the adjusted bounding contour
    M = cv.moments(bounding_contour)
    
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"]) + x  # Add x offset back to centroid
        cy = int(M["m01"] / M["m00"]) + y  # Add y offset back to centroid
    else:
        cx, cy = 0, 0
    
    return cx, cy


def debug_show_contours(base_image : Image,contours : typing.Sequence[cv2.typing.MatLike], fill = True) -> None:
    img = base_image.deep_copy().to_rgb()
    
    for i, contour in enumerate(contours):
        cx, cy = calculate_bounding_centroid(contour)

        color = np.random.randint(0, 255, size=(3,))
        if (fill):
            cv.drawContours(img.data, [contour], -1, color.tolist(), cv.FILLED)
        cv.drawContours(img.data, [contour], -1, (0, 0, 255), 1)
        write_stroked_text(img.data, str(i), cx - 10  , cy)

    img.display()

def debug_show_bboxes(base_image : Image, bboxes: list):
    imgCopy = base_image.deep_copy()
    for bbox in bboxes:
        x, y, w, h = bbox
        cv.rectangle(imgCopy.data, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    imgCopy.display()

def debug_show_bboxes_dict(base_image: Image, key_rects : dict):
    img_copy = base_image.deep_copy()

    for key_number, rect in key_rects.items():
        x, y, w, h = rect
        cv.rectangle(img_copy.data, (x, y), (x + w, y + h), (0, 255, 0), 2)

    img_copy.display()

def create_mask_from_contours(image_shape : tuple, contours : typing.Sequence[cv2.typing.MatLike]):
    mask = np.zeros(image_shape, dtype=np.uint8)
    cv.drawContours(mask, contours, -1, (255, 255, 255), thickness=cv.FILLED)
    
    return mask


def reverse_mask(mask : cv2.typing.MatLike):
    # Create a white canvas with the same shape as the mask
    reversed_mask = np.ones_like(mask, dtype=np.uint8) * 255
    
    # Subtract the mask from the white canvas
    reversed_mask = cv.bitwise_and(reversed_mask, np.array(255) - mask)
    
    return reversed_mask


def close_gaps(mask : cv2.typing.MatLike, kernel_size=5):
    # Define a kernel for morphological operations
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Perform dilation followed by erosion to close gaps
    closed_mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    
    return closed_mask

def merge_and_sort_contours(contours1 : typing.Sequence[cv2.typing.MatLike], contours2 : typing.Sequence[cv2.typing.MatLike]):
    # Concatenate the two tuples of contours
    
    all_contours = []
    all_contours += contours1
    all_contours += contours2
    
    # Sort the contours based on the x-coordinate of their bounding rectangles
    sorted_contours = sorted(all_contours, key=lambda c: cv.boundingRect(c)[0])
    
    return sorted_contours




def extract_key_bounding_rects(contours : typing.Sequence[cv2.typing.MatLike]):
    key_rects = {}  # Initialize an empty dictionary to store key bounding rects

    for i, contour in enumerate(contours):
        x, y, w, h = cv.boundingRect(contour)
        key_rects[i] = (x, y, w, h)  # Store the bounding rect in the dictionary with key number as the key

    return key_rects


#https://en.wikipedia.org/wiki/Shoelace_formula
def triangle_area(p1, p2, p3):
    if len(p1) < 2 or len(p2) < 2 or len(p3) < 2:
        return 0  # Return 0 if any of the points does not have both x and y coordinates
    return abs((p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1])) / 2)


def simplify_contours_to_vertices(contours : typing.Sequence[cv2.typing.MatLike], num_vertices):
    simplified_contours = []
    for contour in contours:
        areas = np.zeros(len(contour))
        for i in range(1, len(contour) - 1):
            areas[i] = triangle_area(contour[i - 1], contour[i], contour[i + 1])

        sorted_indices = np.argsort(areas)[::-1]

        simplified_contour = np.array(contour)[sorted_indices[:num_vertices]]
        simplified_contours.append(simplified_contour)

    return simplified_contours

def simplify_contours_douglas_peucker(contours : typing.Sequence[cv2.typing.MatLike], epsilon = 5):
    simplified_contours = [cv.approxPolyDP(contour, epsilon, True) for contour in contours]
    return simplified_contours



def contours_to_list(contours : typing.Sequence[cv2.typing.MatLike]):
    contour_list = []
    for contour in contours:
        points = contour.squeeze().tolist()  
        contour_list.append(points)
    return contour_list

def get_key_color(image: Image, contour : cv2.typing.MatLike):
    mask = np.zeros(image.data.shape[:2], dtype=np.uint8)
    cv.drawContours(mask, [contour], -1, [255], cv.FILLED)
    # Image(mask).display()
    mean = cv.mean(image.data, mask=mask)
    return mean



def get_keys_colors(image: Image, regions: list) -> list:
    colors = []
    for region in regions:
        average_color = get_key_color(image, np.mat(region))
        colors.append(average_color)
    return colors



def detect_key_presses(curr_colors: list[tuple], base_colors: list[tuple], threshold: int = 20) -> list:
    key_presses = []

    for curr_color, base_color in zip(curr_colors, base_colors):
        distance = np.linalg.norm(np.array(curr_color) - np.array(base_color))
        
        if int(distance) > threshold:
            key_presses.append(True)  # pressed
        else:
            key_presses.append(False)  # not pressed

    return key_presses


def read_video(video_path: str, action : typing.Callable, frame_interval: int = 1, start = 0 , end = None) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Couldn't open the video file.")
        return

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
    cap.release()




def convert_bool_to_midi(frames_per_second: int, beats_per_minute: int , notes_state_frame_map: dict[int, list[bool]], output_file: str) -> None:
    # tempo = mido.bpm2tempo(beats_per_minute)
    bpm = 124
    mspb = mido.bpm2tempo(bpm) #tempo
    tpb = 480
    midi_file = mido.MidiFile(ticks_per_beat=tpb)
    track = mido.MidiTrack()
    tempo_meta = MetaMessage('set_tempo', tempo=mspb)
    track.append(tempo_meta)

    current_notes = set()
    ticks_since_last_msg = 0
    for frame, note_states in notes_state_frame_map.items():
        print(frame)
        for note, state in enumerate(note_states):
            print(note, state,ticks_since_last_msg)
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


___image = cv.imread("baseFull.png")
assert ___image is not None, "file could not be read, check with os.path.exists()"


complexImage = Image(___image)


roi_ready_image = complexImage.deep_copy().to_grayscale().threshold(150,200).get_canny_edges()
roi_contours = roi_ready_image.get_contours()

bboxes = getbboxes_dbscan(flatmap_contour_points(roi_contours))
target_bbox_roi = get_largest_bbox(bboxes)

# debug_show_bboxes(complexImage.deep_copy().to_rgb(),[target_bbox_roi])

roi_image = complexImage.deep_copy().crop(*target_bbox_roi).crop_around(10)

keys_contour_img = roi_image.deep_copy().to_grayscale().threshold(190,255)
white_keys_contours = keys_contour_img.get_contours()

# debug_show_contours(roi_image,white_keys_contours)

contoured_mask = create_mask_from_contours(roi_image.data.shape,white_keys_contours)
not_contoured_mask = Image(reverse_mask(close_gaps(contoured_mask, 10)))

black_keys_contours = not_contoured_mask.deep_copy().to_grayscale().get_contours()
# debug_show_contours(roi_image,black_keys_contours)

all_keys_contours = merged_and_sorted_contours = merge_and_sort_contours(white_keys_contours, black_keys_contours)
simplified_contours = simplify_contours_douglas_peucker(all_keys_contours,8)

# debug_show_contours(roi_image,simplified_contours)

# bounding_rects = extract_key_bounding_rects(simplified_contours)

key_regions = contours_to_list(simplified_contours)

base_average_colors = get_keys_colors(roi_image, key_regions)


playback_frames={}

def action(frameNumber: int,frame:cv2.typing.MatLike):
    frame_img = Image(frame)
    frame_img.crop(*target_bbox_roi).crop_around(10)
    frame_colors = get_keys_colors(frame_img,key_regions)
    
    keypresses = detect_key_presses(frame_colors,base_average_colors)
    print(f'processed frame {frameNumber}')
    print(f'keys pressed : {[i for i,kp in enumerate(keypresses) if kp == True]}')
    playback_frames[frameNumber] = keypresses
    


AYAYA=7598

# read_video("src.mp4",action,1,AYAYA,10004)

file_path = 'key_value_data.pkl'

# with open(file_path, 'wb') as f:
#     pickle.dump(playback_frames, f)

# frames_to_midi(60,124,playback_frames)
    
with open(file_path, 'rb') as f:
    loaded_kv_data = pickle.load(f)

# Now loaded_kv_data contains the same key-value pairs as kv_data
print("Loaded key-value pairs")
generated_midi = convert_bool_to_midi(60, 124,loaded_kv_data,'output.mid')