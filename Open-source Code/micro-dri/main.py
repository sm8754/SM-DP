import serial.tools.list_ports
import os
import sys
from AFocus import AFocus
import time
import warnings
import cv2
from Analysis import Predictor
import tkinter as tk
from tkinter import ttk
from copy import deepcopy
from threading import Thread
import sys
import numpy as np
import glob
from PIL import Image

warnings.filterwarnings("ignore")
sys.path.append(os.getcwd())
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class RedirectedOutput:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, string):
        self.text_widget.insert(tk.END, string)
        self.text_widget.see(tk.END)

    def flush(self):
        pass

def create_window():
    window = tk.Tk()
    window.title("real-time result")
    window.geometry("600x400")
    text_output = tk.Text(window, wrap=tk.WORD, font=("Arial Unicode MS", 10))
    text_output.pack(expand=True, fill=tk.BOTH)
    sys.stdout = RedirectedOutput(text_output)
    window.mainloop()

def mv(s):

    pos10 = [0, 780000]
    pos11 = [1900000, 3300000]
    pos20 = [2600000, 780000]
    pos21 = [4750000, 3300000]
    pos30 = [5400000, 780000]
    pos31 = [7600000, 3300000]
    pos40 = [8400000, 780000]
    pos41 = [10500000, 3300000]

    pos_start = [pos10, pos20, pos30, pos40]
    pos_end = [pos11, pos21, pos31, pos41]
    if s == 1:
        return pos10, pos11
    if s == 2:
        return pos20, pos21
    if s == 3:
        return pos30, pos31
    if s == 4:
        return pos40, pos41
    if s == 0:
        return pos_start, pos_end

def prepare_pos(ser, pos):
    data = str('Control command for stage ') + str(pos[0]) +str(',') + str(pos[1]) + str(',Initial_focus') + str('\r')
    print("Initial coordinates of the stage:  %s " % data)
    write_len = ser.write(data.encode('utf-8'))
    t0 = time.time()
    while True:
        com_input = ser.read(20)
        t1 = time.time()
        t = t1 - t0
        if com_input or t >= 5:
            if com_input:
                print("\n%s" % com_input)
            else:
                print("\n%s" % "Nothing received")
            break

def converter(ser, wjtn):

    if wjtn == 4:
        ratio = "4 X"
        kongwei = 3     #'The hole position where the objective lens is located'
    elif wjtn == 10:
        ratio = "10 X"
        kongwei = 4
    elif wjtn == 20:
        ratio = "20 X"
        kongwei = 5
    elif wjtn == 40:
        ratio = "40 X"
        kongwei = 1
    else:
        kongwei = 3
        ratio = "4 X"
        print('The expected magnification is: [4 X, 10 X, 20 X, 40 X]. 4 X is selected by default.')

    data = str('Control command for objective lens ') + str(kongwei) + str('\r')
    a = ser.write(data.encode('utf-8'))
    t0 = time.time()
    while True:
        com_input = ser.read(20)
        t1 = time.time()
        t = t1 - t0
        if com_input or t >= 5:
            if com_input:
                # print("\n%s" % com_input)
                print()
            else:
                print("\n%s" % "Nothing received")
            del data
            break

    print("Conversion completed!\n")
    print("The magnification of object lens is:  %s \n" % ratio)

def step(wjtn):

    step_4x = [370000, 245000]
    step_10x = [150000, 100000]
    step_20x = [75000, 50000]
    step_40x = [37500, 25000]

    if wjtn == 4:
        return step_4x
    elif wjtn == 10:
        return step_10x
    elif wjtn == 20:
        return step_20x
    elif wjtn == 40:
        return step_40x

def sled(ser, val):
    data = str('Control command for LED ') + str(val) + str('\r')
    print(val)
    write_len = ser.write(data.encode('utf-8'))
    t0 = time.time()
    while True:
        com_input = ser.read(20)
        t1 = time.time()
        t = t1 - t0
        if com_input or t >= 5:
            if com_input:
                print("\n%s" % com_input)
            else:
                print("\n%s" % "Nothing received")
            break

def find_top_left_focus(pos0, pos1, current_X, current_Y, focus_coords, stepX, stepY):
    grid_x = (current_X - pos0[0]) // stepX
    grid_y = (current_Y - pos0[1]) // stepY
    grid_x = max(1, min(grid_x, Focus_time_X - 2))
    grid_y = max(1, min(grid_y, Focus_time_Y - 2))
    for focus in focus_coords:
        if (focus[0] == pos0[0] + grid_x * stepX) and (focus[1] == pos0[1] + grid_y * stepY):
            return focus
    return None

def get_corners_focus_values(top_left_focus, focus_coords, stepX, stepY):
    top_left = top_left_focus
    top_right = bottom_left = bottom_right = None
    for focus in focus_coords:
        if focus[1] == top_left[1] and focus[0] == top_left[0] + stepX:
            top_right = focus
        elif focus[0] == top_left[0] and focus[1] == top_left[1] + stepY:
            bottom_left = focus
        elif focus[0] == top_left[0] + stepX and focus[1] == top_left[1] + stepY:
            bottom_right = focus

    return top_left, top_right, bottom_left, bottom_right


def adaptive_median_filter(Z_values, max_size=3):

    height, width = Z_values.shape
    smoothed_Z = np.copy(Z_values)

    for y in range(height):
        for x in range(width):
            size = min(max_size, y, height - y - 1, x, width - x - 1)
            size = max(1, size)
            half_size = size // 2
            window = Z_values[max(y - half_size, 0):min(y + half_size + 1, height),
                              max(x - half_size, 0):min(x + half_size + 1, width)]
            smoothed_Z[y, x] = np.median(window)

    return smoothed_Z

def Splicing_hcc(wj, filepath):

    img_folders = glob.glob(filepath + "\\_mask.png")
    MAX_X = -1
    MAX_Y = -1
    for img_path in img_folders:
        filename = img_path.split('\\')[-1]
        x, y = map(int, filename.split('.')[0].split('_')[1:3])
        MAX_X = max(MAX_X, x)
        MAX_Y = max(MAX_Y, y)
    step_4x = ['steps for stage']
    step_10x = ['steps for stage']
    step_20x = ['steps for stage']
    step_40x = ['steps for stage']
    if wj == 10:
        gap_x = step_10x[0]
        gap_y = step_10x[1]
    elif wj == 20:
        gap_x = step_20x[0]
        gap_y = step_20x[1]
    IMAGE_ROW = MAX_Y // gap_y + 1
    IMAGE_COLUMN = MAX_X // gap_x + 1
    IMAGE_SIZE_X = 'image_size'
    IMAGE_SIZE_Y = 'image_size'

    to_image = Image.new('RGB', ((IMAGE_COLUMN - 1) * IMAGE_SIZE_X, IMAGE_ROW * IMAGE_SIZE_Y))
    for img_path in img_folders:
        filename = img_path.split('\\')[-1]
        x, y = map(int, filename.split('.')[0].split('_')[1:3])
        i = x // gap_x
        j = y // gap_y
        from_image = Image.open(img_path).resize((IMAGE_SIZE_X, IMAGE_SIZE_Y), Image.ANTIALIAS)
        to_image.paste(from_image, (i * IMAGE_SIZE_X, j * IMAGE_SIZE_Y))
    final_image_path = filepath + "name.png"
    to_image.save(final_image_path)
    print("Image stitched and saved at", final_image_path)
    return final_image_path

def find_foreground_rect(image_path, microscope_start, microscope_end):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: No image fond")
        return None
    edged = cv2.Canny(image, 50, 255)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)
    true_x, true_y, true_w, true_h = x * 8, y * 8, w * 8, h * 8

    image_width, image_height = image.shape[:2]

    microscope_width = microscope_end[0] - microscope_start[0]
    microscope_height = microscope_end[1] - microscope_start[1]

    microscope_per_pixel_x = microscope_width / image_width
    microscope_per_pixel_y = microscope_height / image_height

    microscope_coord_start = [microscope_start[0] + true_x * microscope_per_pixel_x, microscope_start[1] + true_y * microscope_per_pixel_y]
    microscope_coord_end = [microscope_end[0] + true_x * microscope_per_pixel_x, microscope_end[1] + true_y * microscope_per_pixel_y]

    return microscope_coord_start, microscope_coord_end

def find_tissue_under_low_magnification(ser, cap, analyzer, pos10, pos11):
    first_focus = 955500
    stepn = step(10)
    step_x = stepn[0]
    step_y = stepn[1]
    current_posx = pos10[0]
    current_posy = pos10[1]
    path = '/TDF/temp'
    while (current_posy <= pos11[1]):
        k = 0
        move_xyz(ser, current_posx, current_posy, first_focus, 1)
        clear_buffer(cap, 1)
        time.sleep(0.05)
        while (current_posx <= pos11[0]):
            k = k + 1
            name = 'IMG_' + str(current_posx - pos10[0]) + '_' + str(current_posy - pos10[1])
            current_posx = current_posx + step_x
            move_xyz(ser, current_posx, current_posy, first_focus, 0)
            clear_buffer(cap, 1)
            time.sleep(0.01)
            AFocus(ser, cap, current_posx, current_posy, first_focus, fine_flag=False)
            img_path = photo_a_microview(cap, path, name)
            analyzer.predict(10, img_path)
        current_posx = pos10[0]
        current_posy = current_posy + step_y
    Splicing_path = Splicing_hcc(10, filepath=path)
    start_point_, end_point_ = find_foreground_rect(Splicing_path, microscope_start = pos10, microscope_end = pos11)
    return start_point_, end_point_

def autofocus_and_record_positions(ser, cap, analyzer, pos10, pos11, Focus_time_X, Focus_time_Y):
    converter(ser, 10)
    sled(ser, 560)
    start_point_, end_point_ = find_tissue_under_low_magnification(ser, cap, analyzer, pos10, pos11)
    converter(ser, 20)
    sled(ser, 1000)
    focus_positions = []
    steps_X = (end_point_[0] - start_point_[0]) // (Focus_time_X - 1)
    steps_Y = (end_point_[1] - start_point_[1]) // (Focus_time_Y - 1)
    max_clarity = 0
    current_Z = 925000
    for i in range(1, Focus_time_Y-1):
        current_Y = start_point_[1] + i * steps_Y
        clarity_list = []
        z_positions = []
        current_row_indices = []
        for j in range(1, Focus_time_X-1):
            current_X = start_point_[0] + j * steps_X
            move(ser, current_X, current_Y, 1)
            clear_buffer(cap, 1)
            current_Z, max_clarity, frames = AFocus(ser, cap, current_X, current_Y, current_Z, fine_flag = False, max_clarity_former = max_clarity)
            clarity_list.append(max_clarity)
            z_positions.append(current_Z)
            focus_positions.append([current_X, current_Y, current_Z, max_clarity])
            current_row_indices.append(len(focus_positions) - 1)
        for index in current_row_indices:
            if (focus_positions[index][3] < 6):
                closest_foreground_index = None
                min_distance = float('inf')
                for other_index in current_row_indices:
                    if (focus_positions[other_index][3] > 8):
                        distance = abs(index - other_index)
                        if distance < min_distance:
                            closest_foreground_index = other_index
                            min_distance = distance

                if closest_foreground_index is not None:
                    focus_positions[index][2] = focus_positions[closest_foreground_index][2]
    return focus_positions, steps_X, steps_Y, start_point_, end_point_


def interpolate_focus(x_percent, y_percent, top_left, top_right, bottom_left, bottom_right):

    top_focus = top_left + (top_right - top_left) * x_percent
    bottom_focus = bottom_left + (bottom_right - bottom_left) * x_percent
    focus_value = top_focus + (bottom_focus - top_focus) * y_percent

    return focus_value

def clear_buffer(cap, frames_to_clear=2):
    for _ in range(frames_to_clear):
        frame = cap.run()

def photo_a_microview(cap, save_path, name):
    frame = cap.run()
    cv2.namedWindow("capture", 0)
    cv2.resizeWindow("capture", 600, 400)
    cv2.imshow("capture", frame)
    cv2.waitKey(1)
    image_filename = f'{name}.png'
    cv2.imwrite(save_path + image_filename, frame)

    return save_path + image_filename

class Camera_Collecting():

    def __init__(self):
        'Initialize collection device'

    def run(self, autoExpo=False, autoFocus=False, autoWB=False, FfcOnce=False, ser=None):
        'excute and set'

        return self.img

def move(ser, current_posx, current_posy, judge):
    data = str('control command for stage ') + str(current_posx) + str(',') + str(current_posy) + str('\r')
    write_len = ser.write(data.encode('utf-8'))
    if judge == 1:
        t0 = time.time()
        while True:
            com_input = ser.read(20)
            t1 = time.time()
            t = t1 - t0
            if com_input or t >= 5:
                if com_input:
                    a = 0
                else:
                    print("\n%s" % "Nothing received")
                break

def move_xyz(ser, current_posx, current_posy, focus_z, judge):
    data = str('control command for stage ') + str(current_posx) + str(',') + str(current_posy) + str(',') + str(focus_z) + str('\r')

    write_len = ser.write(data.encode('utf-8'))
    if judge == 1:
        t0 = time.time()
        while True:
            com_input = ser.read(20)
            t1 = time.time()
            t = t1 - t0
            if com_input or t >= 5:
                if com_input:
                    a = 0
                else:
                    print("\n%s" % "Nothing received")
                break


if __name__ == '__main__':
    Thread(target=create_window).start()
    ports_list = list(serial.tools.list_ports.comports())
    if len(ports_list) <= 0:
        print("no device.")
    else:
        print("\nThe following serial ports are available: \n")
        print("%-10s %-50s %-10s" % ("num", "name", "number"))
        for i in range(len(ports_list)):
            comport = list(ports_list[i])
            comport_number, comport_name = comport[0], comport[1]
            print("%-10s %-50s %-10s" % (i, comport_name, comport_number))

        port_num = ports_list[0][0]
        print("The default selection for serial ports is: %s\n" % port_num)
        ser = serial.Serial(port=port_num, baudrate=9600, bytesize=serial.SEVENBITS, stopbits=serial.STOPBITS_TWO,
                            timeout=0.5)
        if ser.isOpen():
            print("Successfully opened serial port, serial port number: %s" % ser.name)
        else:
            print("Unable to open serial port.")
        print('\n')

    cap = Camera_Collecting()
    analyzer = Predictor()
    positive_num = 0
    wjtn = 10
    fine_flag = False
    slide_num = [1, 2, 3, 4]
    Focus_time_X, Focus_time_Y = 6, 13
    for s in slide_num:
        pos_0, pos_1 = mv(s)
        stepn = step(wjtn)
        step_x = stepn[0]
        step_y = stepn[1]
        focus_coords, steps_X, steps_Y, start_point_, end_point_ = autofocus_and_record_positions(ser, cap, analyzer, pos_0, pos_1, Focus_time_X, Focus_time_Y)
        current_posx = start_point_[0]  # pos0 = [10500000, 3300000]    pos1 = [8400000, 780000]
        current_posy = start_point_[1]
        path = '/slide' + str(s) + '_' + str(wjtn) + 'x_0' + '/'
        if not os.path.exists(path):
            os.mkdir(path)
        pos_flag = 0
        while (current_posy <= end_point_[1]):
            k = 0
            top_left_focus = find_top_left_focus(start_point_, end_point_, current_posx, current_posy, focus_coords, steps_X, steps_Y)
            if top_left_focus:
                top_left, top_right, bottom_left, bottom_right = get_corners_focus_values(top_left_focus, focus_coords,
                                                                                          steps_X, steps_Y)
                if top_right is None or bottom_left is None:
                    focus_Z = top_left_focus[2]
                else:
                    x_relative = (current_posx - top_left_focus[0]) // steps_X
                    y_relative = (current_posy - top_left_focus[1]) // steps_Y
                    focus_Z = interpolate_focus(
                        x_relative, y_relative,
                        top_left[2], top_right[2],
                        bottom_left[2], bottom_right[2],
                        )
                move_xyz(ser, current_posx, current_posy, focus_Z, 1)
            clear_buffer(cap, 1)
            while (current_posx <= end_point_[0]):
                k = k + 1
                name = 'IMG_' + str(current_posx - start_point_[0]) + '_' + str(current_posy - start_point_[1])
                current_posx = current_posx + step_x
                top_left_focus = find_top_left_focus(start_point_, end_point_, current_posx, current_posy, focus_coords, steps_X, steps_Y)
                if top_left_focus:
                    top_left, top_right, bottom_left, bottom_right = get_corners_focus_values(top_left_focus, focus_coords,
                                                                                              steps_X, steps_Y)

                    if top_right is None or bottom_left is None:
                        focus_Z = top_left_focus[2]
                    else:
                        x_relative = (current_posx - top_left_focus[0]) // steps_X
                        y_relative = (current_posy - top_left_focus[1]) // steps_Y

                        focus_Z = interpolate_focus(
                            x_relative, y_relative,
                            top_left[2], top_right[2],
                            bottom_left[2], bottom_right[2],
                            )
                    move_xyz(ser, current_posx, current_posy, focus_Z, 0)
                    clear_buffer(cap, 1)
                    photo_a_microview(cap, path, name)
                    analyzer.predict(20, path)
            current_posx = start_point_[0]
            current_posy = current_posy + step_y
        Splicing_path = Splicing_hcc(20, filepath=path)
        cv2.destroyAllWindows()
    ser.close()
    cv2.destroyAllWindows()



