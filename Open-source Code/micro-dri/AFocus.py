
# import camera
import time
import serial.tools.list_ports
import cv2
import os
import shutil
import numpy as np
import math
from skimage import data, filters

down_z = 3248400
up_z = 3259000
step_z = 200
step_z_fine = 100

save_path = '/AFocus'
save_temp_af_imgs = False
# Used for camera initialization and returning one frame of FOV
class Camera_Collecting():

    def __init__(self):
        'Initialize collection device'

    def run(self, autoExpo=False, autoFocus=False, autoWB=False, FfcOnce=False, ser=None):
        'excute and set'

        return self.img

def Folder_count(path):
    count = 0
    filelist = os.listdir(path)
    for f in filelist:
        filepath = os.path.join(path, f)
        if os.path.isfile(filepath):
            os.remove(filepath)
        elif os.path.isdir(filepath):
            shutil.rmtree(filepath, True)
    for file in os.listdir(path):
        file = os.path.join(path, file)
        if os.path.isdir(file):
            count = count+1
    return count

def move_z(ser, current_Z, wait):

    data = str('Control command for z stage ') + str(current_Z) + str('\r')
    ser.write(data.encode('utf-8'))
    t0 = time.time()
    while wait:
        com_input = ser.readall()
        t1 = time.time()
        t = t1 - t0
        if com_input or t >= 5:
            if com_input:
                return True

def Laplacian(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray_img, cv2.CV_64F).var()

def tenengrad(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out = filters.sobel(gray_img)
    out = np.sum(out**2)
    out = np.sqrt(out)
    return out

def Compute_Clarity(img, save_flag):
    if save_flag:
        img = cv2.imread(img, cv2.IMREAD_COLOR)
    clarity = Laplacian(img)
    return clarity

def clear_buffer(cap, frames_to_clear=3):
    for _ in range(frames_to_clear):
        # ret, frame = cap.read()
        frame = cap.run()

def photo_a_microview(cap, current_X, current_Y, current_Z, save_flag, qq, show_focus = True):

    frame = cap.run()
    if save_flag:
        if qq ==1:
            image_filename = f'{int(current_X/100)}_{int(current_Y/100)}_{int(current_Z/100)}.bmp'
        else:
            image_filename = f'{int(current_X/100)}_{int(current_Y/100)}_{int(current_Z/100)}.png'
        cv2.imwrite(save_path + image_filename, frame)
        out = Compute_Clarity(save_path + image_filename, save_flag)
    else:

        out = Compute_Clarity(frame, save_flag)

    if current_Z <= up_z and (not show_focus):
        window_name = "Capture Window"
        cv2.namedWindow(window_name, 0)
        cv2.resizeWindow(window_name, 625, 500)
        cv2.imshow(window_name, frame)
        cv2.setWindowTitle(window_name, f"X: {current_X}, Y: {current_Y}, Z: {current_Z}, Clarity: {out}")
        cv2.waitKey(1)
    elif show_focus:
        clear_buffer(cap, 1)
        frame = cap.run()
        cv2.destroyWindow("Capture Window")
        window_name = "Focus Window"
        cv2.namedWindow(window_name, 0)
        cv2.resizeWindow(window_name, 625, 500)
        cv2.imshow(window_name, frame)
        cv2.setWindowTitle(window_name, f"X: {current_X}, Y: {current_Y}, Clarity: {out}")
        cv2.waitKey(1)

    return out, frame

def Fine_focus(ser, cap, current_X, current_Y, Focus_point_Z):
    Folder_count(save_path)
    clarity_list = []
    range_down = Focus_point_Z-step_z
    range_up = Focus_point_Z + step_z
    for current_Z in range(range_down, range_up, step_z_fine):
        move_z(ser, current_Z, 0)
        time.sleep(0.05)
        clarity, frames = photo_a_microview(cap, current_X, current_Y, current_Z, save_flag=save_temp_af_imgs,qq=0)
        record_every_position = {'position': current_Z, 'clarity': clarity}
        clarity_list.append(record_every_position)
    max_clarity = 0
    for index in clarity_list:
        if index['clarity'] > max_clarity:
            max_clarity = index['clarity']
            Focus_point_Z_new = int(index['position'])
    return Focus_point_Z_new, max_clarity

def AFocus(ser, cap, current_X, current_Y, current_Z_former, fine_flag = False, max_clarity_former = 0):
    clarity_list = []
    decline_count = 0
    clear_buffer(cap, 1)
    if max_clarity_former > 15:
        down_z = current_Z_former - 10 * step_z
        up_z = current_Z_former + 10 * step_z
    else:
        down_z = 3248400
        up_z = 3259000
    for current_Z in range(down_z, up_z, step_z):

        move_z(ser, current_Z, 0)
        clear_buffer(cap, 1)
        clarity, frames = photo_a_microview(cap, current_X, current_Y, current_Z, save_flag=save_temp_af_imgs, qq=0, show_focus = False)
        record_every_position = {'position': current_Z, 'clarity': clarity}
        clarity_list.append(record_every_position)
        if len(clarity_list) > 1 and clarity_list[-1]['clarity'] < clarity_list[-2]['clarity']:
            decline_count += 1
        else:
            decline_count = 0
        if decline_count >= 5:
            break

    max_clarity = 0
    time.sleep(0.1)
    index_flag = 0
    for index in clarity_list[1:-1]:
        index_flag = index_flag + 1
        if index['clarity'] > max_clarity:
            max_clarity = index['clarity']
            Focus_point_Z = index['position']
            index_flags  = index_flag

    if fine_flag:
        move_z(ser, Focus_point_Z-step_z, 0)
        Focus_point_Z, max_clarity = Fine_focus(ser, cap, current_X, current_Y, Focus_point_Z)
    move_z(ser, Focus_point_Z, 1)
    time.sleep(0.05)
    clear_buffer(cap, 1)
    clarity, _ = photo_a_microview(cap, current_X, current_Y, Focus_point_Z, save_flag=False, qq=1, show_focus = True)

    return Focus_point_Z, max_clarity, frames

if __name__ == "__main__":
    ports_list = list(serial.tools.list_ports.comports())
    if len(ports_list) <= 0:
        print("no device.")
    else:
        print("The following serial ports are available:\n")
        print("%-10s %-50s %-10s" % ("num", "name", "number"))
        for i in range(len(ports_list)):
            comport = list(ports_list[i])
            comport_number, comport_name = comport[0], comport[1]
            print("%-10s %-50s %-10s" % (i, comport_name, comport_number))
        port_num = ports_list[0][0]
        print("The default selection for serial ports is: %s\n" % port_num)
        ser = serial.Serial(port='COM4', baudrate=9600, bytesize=serial.SEVENBITS, stopbits=serial.STOPBITS_TWO,
                            timeout=0.5)
        if ser.isOpen():
            print("Successfully opened serial port, serial port number: %s" % ser.name)
        else:
            print("Unable to open serial port.")
        print('\n')
    collect = Camera_Collecting()
    save_path = '/AFocus'
    save_temp_af_imgs = False
    AFocus(ser, collect, 6338451, 1094571, 3251000, fine_flag=False)

