# -*- coding=utf-8 -*-
import os
import tkinter as tk
from tkinter.ttk import *
from tkinter.filedialog import *
from PIL import Image, ImageTk
import time
from vector_image.trans import *
import re
import threading

lock = threading.RLock()


def show(cur_img):
    global canvas, imgTK
    imgPIL = Image.fromarray(cur_img)
    imgTK = ImageTk.PhotoImage(image=imgPIL)
    canvas.create_image(594, 420, anchor='center', image=imgTK)


def use_dilate_and_erode():
    return dae_state_var == dae_states_list[0]


def is_equal(params, other_params):
    if params == other_params:
        return True
    elif params[9] != other_params[9]:
        return False
    elif params[0] is False and other_params[0] is False:
        if params[3] is False and other_params[3] is False:
            if params[6] == 'Not Use' and other_params[6] == 'Not Use':
                return True
            elif params[6] == 'After' and other_params[6] == 'After':
                return True
            elif params[6:] == other_params[6:]:
                return True
            else:
                return False
        elif params[3:] == other_params[3:]:
            return True
        else:
            return False
    return False


annotation = [
    '# This file is an auto output of the vector image convert program\n',
    '# The following data will be arranged like this:\n',
    '# X coefficients\n',
    '# Y coefficients\n',
    '# Segment Length\n',
    '# which are headed by the division of the Generated Image Min Size between the Aimed Image Min Size\n',
    '# (default to be 210x297, A4 paper size).\n',
    '# For example, if I generate the vector image from a bitmap of 840x1188, then the param will be 4\n',
    '# (840 / 210) = 4.\n',
    '# This file will be end with -1\n'
]


def roll_polling():
    global cur_params, last_params, last_stable_params, img, origin_btn_20, is_watching
    global img_vector, img_contour, img_origin, cur_fd_params, cur_generate_params
    while 1:
        # cur_params[]
        cur_params[0] = use_gaussian_blur.get()
        cur_params[1] = gaussian_kernel_size.get()
        cur_params[2] = gaussian_mean_value.get()
        cur_params[3] = use_mean_shift.get()
        cur_params[4] = mean_shift_sp.get()
        cur_params[5] = mean_shift_sr.get()
        cur_params[6] = dae_state_var.get()
        cur_params[7] = dae_kernel_size_bf.get()
        cur_params[8] = dae_iter_nums_bf.get()
        if cluster_nums.get() == '':
            cur_params[9] = 4
        else:
            cur_params[9] = int(cluster_nums.get())

        cur_fd_params[0] = dae_state_var.get()
        cur_fd_params[1] = dae_kernel_size_at.get()
        cur_fd_params[2] = dae_iter_nums_at.get()

        if seg_length.get() == '':
            cur_generate_params[0] = 20
        else:
            cur_generate_params[0] = float(seg_length.get())
        if step.get() == '':
            cur_generate_params[1] = 2
        else:
            cur_generate_params[1] = float(step.get())

        if is_equal(cur_params, last_stable_params):
            time.sleep(1)
            continue

        if not is_equal(cur_params, last_stable_params):
            last_params = []
            for param in cur_params:
                last_params.append(param)
            while 1:
                time.sleep(1)
                cur_params[0] = use_gaussian_blur.get()
                cur_params[1] = gaussian_kernel_size.get()
                cur_params[2] = gaussian_mean_value.get()
                cur_params[3] = use_mean_shift.get()
                cur_params[4] = mean_shift_sp.get()
                cur_params[5] = mean_shift_sr.get()
                cur_params[6] = dae_state_var.get()
                cur_params[7] = dae_kernel_size_bf.get()
                cur_params[8] = dae_iter_nums_bf.get()
                if cluster_nums.get() == '':
                    cur_params[9] = 4
                else:
                    cur_params[9] = int(cluster_nums.get())
                if is_equal(cur_params, last_params):
                    last_stable_params = []
                    for param in cur_params:
                        last_stable_params.append(param)
                    break
                else:
                    last_params = []
                    for param in cur_params:
                        last_params.append(param)
        if img is None:
            continue

        is_found = False
        for ind, params in enumerate(past_params_list):
            print(params)
            print(cur_params)
            print(past_params_list)
            if is_equal(cur_params, params):
                img = past_image_list.pop(ind)
                past_params_list.pop(ind)
                print(ind)
                past_image_list.append(img.copy())
                past_params_list.append(cur_params.copy())
                is_found = True
                break
        if not is_found:
            last_operation.set('Processing image!')
            gaussian_blur_parse = (use_gaussian_blur.get(), gaussian_kernel_size.get(), gaussian_mean_value.get())

            mean_shift_parse = (use_mean_shift.get(), mean_shift_sp.get(), mean_shift_sr.get())

            dilate_and_erode_parse = (use_dilate_and_erode(), dae_kernel_size_bf.get(), dae_iter_nums_bf.get())

            th = threading.Thread(target=run_preprocess, args=(img_origin, cur_params[9], gaussian_blur_parse,
                                                               mean_shift_parse, dilate_and_erode_parse))
            th.start()
            th.join()

            past_image_list.append(img.copy())
            past_params_list.append(cur_params.copy())
            if len(past_image_list) > past_image_size:
                past_image_list.pop(0)
                past_params_list.pop(0)

        while 1:
            if is_watching == 1:
                img_vector = None
                img_contour = None
                break
            elif is_watching == 2:
                img_vector = None
                img_contour = None
                show(img)
                break
            elif is_watching == 3:
                img_vector = None
                if img_contour is None:
                    break
                time.sleep(0.5)
            elif is_watching == 4:
                img_contour = None
                if img_vector is None:
                    break
                time.sleep(0.5)


def run_contour(cur_img, dilate_and_erode_parse=(False, 1, 1)):
    lock.acquire()
    global img_contour, last_operation, is_watching
    img_contour = contour(cur_img, dilate_and_erode_parse)
    is_watching = 3
    lock.release()
    show(img_contour)
    last_operation.set('Find contour')


def run_generate(cur_img, cur_seg_length, cur_step):
    lock.acquire()
    global img_vector, last_operation, coefficients_list, is_watching
    img_vector, coefficients_list = generate(cur_img, cur_seg_length, cur_step)
    is_watching = 4
    lock.release()
    show(img_vector)
    last_operation.set('Generate Vector Image')


def run_preprocess(cur_img, cur_cluster_nums, gaussian_params,
                   mean_shift_params, dae_params):
    lock.acquire()
    global img, last_operation, btn_list, past_image_list
    for i, btn in enumerate(btn_list):
        btn['state'] = 'disable'
    img = preprocess(cur_img, cur_cluster_nums, gaussian_params,
                     mean_shift_params, dae_params)
    last_operation.set('Process finished!')
    for j, btn in enumerate(btn_list):
        btn['state'] = 'normal'
    lock.release()


def on_open_btn_12_clicked():
    global img, imgTK, canvas, img_origin, origin_btn_20, img_vector, img_contour, past_image_list, past_params_list
    global is_watching, filename, cur_params
    filepath = askopenfilename(filetypes=[('images(*.png, *.jpg)', '*.png'), ('images(*.png, *.jpg)', '*.jpg')],
                               initialdir='./imgs')
    if len(filepath) > 0:
        img_vector = None
        img_contour = None
        past_image_list = []
        past_params_list = []
        filename.set(filepath)
        last_operation.set('Opened ' + filename.get())
        img = cv.imread(filename.get(), cv.IMREAD_COLOR)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = resize_image(img)
        img_origin = img
        origin_btn_20['text'] = 'Process'
        is_watching = 1
        show(img_origin)

        gaussian_blur_parse = (use_gaussian_blur.get(), gaussian_kernel_size.get(), gaussian_mean_value.get())

        mean_shift_parse = (use_mean_shift.get(), mean_shift_sp.get(), mean_shift_sr.get())

        dilate_and_erode_parse = (use_dilate_and_erode(), dae_kernel_size_bf.get(), dae_iter_nums_bf.get())

        th = threading.Thread(target=run_preprocess, args=(img, cur_params[9], gaussian_blur_parse,
                                                           mean_shift_parse, dilate_and_erode_parse))
        th.start()


def on_origin_btn_20_clicked():
    global img, img_origin, origin_btn_20, is_watching

    if img is not None and img_origin is not None:
        if origin_btn_20['text'] == 'Origin':
            origin_btn_20['text'] = 'Process'
            last_operation.set('Show the original image')
            is_watching = 1
            show(img_origin)
        elif origin_btn_20['text'] == 'Process':
            origin_btn_20['text'] = 'Origin'
            last_operation.set('Show the processed image')
            is_watching = 2
            show(img)


def on_find_contours_btn_clicked():
    global img
    global img_contour, is_watching, last_fd_params, cur_fd_params
    if img_contour is not None and cur_fd_params == last_fd_params:
        last_operation.set('Find contour')
        is_watching = 3
        show(img_contour)
        return
    last_fd_params = cur_fd_params.copy()
    if img is not None:
        dilate_and_erode_parse = (use_dilate_and_erode(), dae_kernel_size_at.get(), dae_iter_nums_at.get())
        last_operation.set('Finding Contour')
        th = threading.Thread(target=run_contour, args=(img, dilate_and_erode_parse))
        th.setDaemon(True)
        th.start()


def on_generate_btn_clicked():
    global img_contour, is_watching, cur_generate_params, last_generate_params
    print(cur_generate_params)
    print(last_generate_params)
    if img_vector is not None and cur_generate_params == last_generate_params:
        last_operation.set('Generate Vector Image')
        is_watching = 4
        show(img_vector)
        return
    if img_contour is not None:
        if cur_generate_params[0] < 10:
            seg_length.set('10')
        elif cur_generate_params[0] > 100:
            seg_length.set('100')
        if cur_generate_params[1] < 0.5:
            step.set('0.5')
        elif cur_generate_params[1] > 10:
            step.set('10')
        cur_generate_params[0] = float(seg_length.get())
        cur_generate_params[1] = float(step.get())
        last_generate_params = cur_generate_params.copy()
        last_operation.set('Generating path')
        th = threading.Thread(target=run_generate, args=(img_contour, cur_generate_params[0], cur_generate_params[1]))
        th.setDaemon(True)
        th.start()


def on_save_btn_clicked():
    global coefficients_list, filename
    file_name = re.split(r'/', filename.get(), 0)[-1]
    if not os.path.exists('./files'):
        os.mkdir('./files')
    print(file_name)
    img_name = file_name.split('.')[0]
    save_path = './files/' + img_name + '.txt'
    with open(save_path, 'w') as f_obj:
        for sentence in annotation:
            f_obj.writelines(sentence)
        f_obj.writelines(str(840 / 210) + '\n')
        for curve_coefficients in coefficients_list:
            for seg_coefficients in curve_coefficients:
                f_obj.writelines(str(seg_coefficients[0][0]) + ',' + str(seg_coefficients[0][1]) + ',' +
                                 str(seg_coefficients[0][2]) + ',' + str(seg_coefficients[0][3]) + '\n')
                f_obj.writelines(str(seg_coefficients[1][0]) + ',' + str(seg_coefficients[1][1]) + ',' +
                                 str(seg_coefficients[1][2]) + ',' + str(seg_coefficients[1][3]) + '\n')
                f_obj.writelines(str(seg_coefficients[2]) + '\n')
            f_obj.writelines('\n')
        f_obj.writelines('-1\n')
        f_obj.close()


def on_pathing_btn_clicked():
    global img_contour
    if step.get() < 0.5:
        step.set(0.5)
    elif step.get() > float(seg_length.get()) / 3:
        step.set(seg_length.get() / 3)
    if img_contour is not None:
        if seg_length.get() < 10:
            seg_length.set(10)
        elif seg_length.get() > 100:
            seg_length.set(100)
        last_operation.set('Generating path')
        th = threading.Thread(target=generate, args=(img_contour, seg_length.get(), step.get()))
        th.setDaemon(True)
        th.start()


def on_options_btn_clicked():
    opt = tk.Toplevel()
    opt.title('Options...')
    opt.geometry('1125x300')
    opt.resizable(False, False)
    last_operation.set('Open Options')


def on_gaussian_blur_checked():
    if use_gaussian_blur.get():
        gks_sb_70['state'] = 'normal'
        gmv_sb_72['state'] = 'normal'
    else:
        gks_sb_70['state'] = 'disabled'
        gmv_sb_72['state'] = 'disabled'
    last_operation.set('Gaussian Blur to ' + str(use_gaussian_blur.get()))


def on_mean_shift_checked():
    if use_mean_shift.get():
        msp_sb_110['state'] = 'normal'
        msr_sb_112['state'] = 'normal'
    else:
        msp_sb_110['state'] = 'disabled'
        msr_sb_112['state'] = 'disabled'
    last_operation.set('Mean Shift to ' + str(use_mean_shift.get()))


def on_dilate_and_erode_changed(var):
    global dae_iter_nums_at, dae_iter_nums_bf
    global dae_kernel_size_at, dae_kernel_size_bf
    if dae_state_var.get() == dae_states_list[1]:
        dks_sb_150['state'] = 'normal'
        din_sb_152['state'] = 'normal'
        dks_sb_170['state'] = 'disabled'
        dae_kernel_size_at.set(3)
        din_sb_172['state'] = 'disabled'
        dae_iter_nums_at.set(1)
    elif dae_state_var.get() == dae_states_list[2]:
        dks_sb_170['state'] = 'normal'
        din_sb_172['state'] = 'normal'
        dks_sb_150['state'] = 'disabled'
        dae_kernel_size_bf.set(3)
        din_sb_152['state'] = 'disabled'
        dae_iter_nums_bf.set(1)
    elif dae_state_var.get() == dae_states_list[3]:
        dks_sb_170['state'] = 'normal'
        din_sb_172['state'] = 'normal'
        dks_sb_150['state'] = 'normal'
        din_sb_152['state'] = 'normal'
        dae_kernel_size_bf.set(3)
        dae_iter_nums_bf.set(1)
        dae_kernel_size_at.set(3)
        dae_iter_nums_at.set(1)
    else:
        dks_sb_150['state'] = 'disabled'
        din_sb_152['state'] = 'disabled'
        dks_sb_170['state'] = 'disabled'
        din_sb_172['state'] = 'disabled'
    last_operation.set('Dilate&Erode to ' + str(dae_state_var.get() != dae_states_list[0]))


root = tk.Tk()
root.title('Curve Former')
root.geometry('1750x900')
root.resizable(False, False)
filename = tk.StringVar()
last_operation = tk.StringVar()
gaussian_kernel_size = tk.IntVar()
gaussian_kernel_size.set(3)
gaussian_mean_value = tk.IntVar()
gaussian_mean_value.set(0)
mean_shift_sp = tk.IntVar()
mean_shift_sp.set(3)
mean_shift_sr = tk.IntVar()
mean_shift_sr.set(100)
dae_kernel_size_bf = tk.IntVar()
dae_kernel_size_at = tk.IntVar()
dae_kernel_size_bf.set(3)
dae_kernel_size_at.set(3)
dae_iter_nums_bf = tk.IntVar()
dae_iter_nums_at = tk.IntVar()
dae_iter_nums_bf.set(1)
dae_iter_nums_at.set(1)
cluster_nums = tk.StringVar()
cluster_nums.set('4')
seg_length = tk.StringVar()
seg_length.set('15')
step = tk.StringVar()
step.set('2')
img = None
img_origin = None
img_contour = None
imgTK = None
img_vector = None
coefficients_list = []
use_gaussian_blur = tk.BooleanVar()
use_mean_shift = tk.BooleanVar()

dae_states_list = ('Not Use', 'Before', 'After', 'Both')
dae_state_var = tk.StringVar()

cur_params = []
cur_fd_params = []
last_fd_params = []
cur_generate_params = []
last_generate_params = []
last_params = []
last_stable_params = []

past_image_size = 10
past_image_list = []
past_params_list = []

is_watching = 0

# ui
canvas_frame = tk.Frame(root, width=1350)
canvas_frame.pack(side='right', fill='y', ipadx=10, ipady=10, expand=0)

param_frame = tk.Frame(root, height=850)
param_frame.pack(side='top', fill='both', ipadx=10, ipady=10, expand=0)

head_frame = LabelFrame(param_frame, text='Operations', height=330)
head_frame.pack(side='top', fill='x', padx=10, pady=10)

sub_param_frame = LabelFrame(param_frame, text='Params', height=600)
sub_param_frame.pack(side='top', fill='x', padx=10, pady=10)

Label(head_frame, text='Select File', width=10).grid(row=1, column=0, columnspan=2, padx=0, pady=15)

file_en_11 = Entry(head_frame, textvariable=filename, state='disabled', width=40)
file_en_11.grid(row=1, column=2, columnspan=8, padx=0, pady=15)

open_btn_12 = Button(head_frame, text='Open...', command=on_open_btn_12_clicked, width=10)
open_btn_12.grid(row=1, column=10, columnspan=2, padx=5, pady=15)

origin_btn_20 = Button(head_frame, text='Process', command=on_origin_btn_20_clicked, width=20)
origin_btn_20.grid(row=2, column=0, columnspan=4, padx=5, pady=20)

contour_btn_21 = Button(head_frame, text='Find Contours', command=on_find_contours_btn_clicked, width=20)
contour_btn_21.grid(row=2, column=4, columnspan=4, padx=5, pady=20)

opt_btn_22 = Button(head_frame, text='Options...', command=on_options_btn_clicked, width=20)
opt_btn_22.grid(row=2, column=8, columnspan=3, padx=5, pady=20)

gen_btn_31 = Button(head_frame, text='Generate', command=on_generate_btn_clicked, width=20)
gen_btn_31.grid(row=3, column=4, columnspan=4, padx=5, pady=20)

save_btn = Button(head_frame, text='Save...', command=on_save_btn_clicked, width=10)
save_btn.grid(row=3, column=10, columnspan=2, padx=5)

btn_list = [origin_btn_20, open_btn_12, contour_btn_21, opt_btn_22, gen_btn_31]

Label(sub_param_frame, text='-----Gaussian Params-----').grid(row=1, column=1)
gauss_chbtn_50 = Checkbutton(sub_param_frame, text='GaussianBlur Enable', variable=use_gaussian_blur, onvalue=True,
                             offvalue=False, command=on_gaussian_blur_checked)
gauss_chbtn_50.grid(row=2, column=1, padx=0, pady=0)

Label(sub_param_frame, text='Kernel Size').grid(row=3, column=0, padx=0, pady=0)
gks_sb_70 = Spinbox(sub_param_frame, textvariable=gaussian_kernel_size, state='disabled', from_=3, to=9, width=10,
                    increment=2)
gks_sb_70.grid(row=4, column=0, padx=0, pady=0)

Label(sub_param_frame, text='Mean').grid(row=3, column=2, padx=0, pady=0)
gmv_sb_72 = Spinbox(sub_param_frame, textvariable=gaussian_mean_value, state='disabled', from_=-3, to=3,
                    increment=0.3, width=10)
gmv_sb_72.grid(row=4, column=2, padx=0, pady=0)

Label(sub_param_frame, text='Mean Shift').grid(row=6, column=1)
ms_chbtn_90 = Checkbutton(sub_param_frame, text='MeanShift Enable', variable=use_mean_shift, onvalue=True,
                          offvalue=False, command=on_mean_shift_checked)
ms_chbtn_90.grid(row=7, column=1, padx=0, pady=10)

Label(sub_param_frame, text='Mean Shift SP').grid(row=8, column=0, padx=0, pady=0)
msp_sb_110 = Spinbox(sub_param_frame, textvariable=mean_shift_sp, state='disabled', from_=1, to=20,
                     increment=1)
msp_sb_110.grid(row=9, column=0, padx=0, pady=0)

Label(sub_param_frame, text='Mean Shift SR').grid(row=8, column=2, padx=0, pady=0)
msr_sb_112 = Spinbox(sub_param_frame, textvariable=mean_shift_sr, state='disabled', from_=50, to=200,
                     increment=9)
msr_sb_112.grid(row=9, column=2, padx=0, pady=0)

Label(sub_param_frame, text='Dilate&Erode').grid(row=10, column=1)

Label(sub_param_frame, text='D&E Enable').grid(row=11, column=1)
dae_cb_131 = Combobox(sub_param_frame, textvariable=dae_state_var, height=1,
                      values=dae_states_list, exportselection=False, state='readonly')
dae_cb_131.bind('<<ComboboxSelected>>', on_dilate_and_erode_changed)
dae_cb_131.grid(row=12, column=1, padx=0, pady=0)

Label(sub_param_frame, text='Kernel Size Bf').grid(row=13, column=0, padx=0, pady=15)
dks_sb_150 = Spinbox(sub_param_frame, textvariable=dae_kernel_size_bf, state='disabled', from_=3, to=9, increment=2)
dks_sb_150.grid(row=14, column=0, padx=0, pady=0)

Label(sub_param_frame, text='Iter Bf').grid(row=13, column=2, padx=0, pady=15)
din_sb_152 = Spinbox(sub_param_frame, textvariable=dae_iter_nums_bf, state='disabled', from_=1, to=9)
din_sb_152.grid(row=14, column=2, padx=0, pady=0)

Label(sub_param_frame, text='Kernel Size At').grid(row=15, column=0, padx=0, pady=15)
dks_sb_170 = Spinbox(sub_param_frame, textvariable=dae_kernel_size_at, state='disabled', from_=3, to=9, increment=2)
dks_sb_170.grid(row=16, column=0, padx=0, pady=0)

Label(sub_param_frame, text='Iter At').grid(row=15, column=2, padx=0, pady=15)
din_sb_172 = Spinbox(sub_param_frame, textvariable=dae_iter_nums_at, state='disabled', from_=1, to=9)
din_sb_172.grid(row=16, column=2, padx=0, pady=0)

Label(sub_param_frame, text='Cluster Nums').grid(row=18, column=1)

Label(sub_param_frame, text='Color Nums').grid(row=19, column=1, padx=0, pady=10)
en41 = Entry(sub_param_frame, textvariable=cluster_nums)
en41.grid(row=20, column=1, padx=0, pady=10)

canvas = tk.Canvas(canvas_frame, bg='white', height=840, width=1188)
canvas.pack(fill='both', pady=10, padx=10)

operate_en_211 = Entry(canvas_frame, textvariable=last_operation, width=20, justify='left', state='disable')
operate_en_211.pack(side='bottom', fill='x', padx=5, pady=5)

Label(sub_param_frame, text='Segment Length').grid(row=21, column=0, padx=0, pady=15)
en42 = Entry(sub_param_frame, textvariable=seg_length)
en42.grid(row=22, column=0, padx=0, pady=10)

Label(sub_param_frame, text='Step').grid(row=21, column=2, padx=0, pady=15)
en43 = Entry(sub_param_frame, textvariable=step)
en43.grid(row=22, column=2, padx=0, pady=10)


###############################################################################


def main():
    global last_stable_params, cur_params
    if use_gaussian_blur.get():
        gks_sb_70['state'] = 'normal'
        gmv_sb_72['state'] = 'normal'

    if use_mean_shift.get():
        msp_sb_110['state'] = 'normal'
        msr_sb_112['state'] = 'normal'

    dae_cb_131.current(0)
    if dae_state_var.get() != dae_states_list[0]:
        dks_sb_150['state'] = 'normal'
        din_sb_152['state'] = 'normal'
        dks_sb_170['state'] = 'normal'
        din_sb_172['state'] = 'normal'

    cur_params.append(use_gaussian_blur.get())
    cur_params.append(gaussian_kernel_size.get())
    cur_params.append(gaussian_mean_value.get())
    cur_params.append(use_mean_shift.get())
    cur_params.append(mean_shift_sp.get())
    cur_params.append(mean_shift_sr.get())
    cur_params.append(dae_state_var.get())
    cur_params.append(dae_kernel_size_bf.get())
    cur_params.append(dae_iter_nums_bf.get())
    cur_params.append(cluster_nums.get())

    cur_fd_params.append(dae_state_var.get())
    cur_fd_params.append(dae_kernel_size_at.get())
    cur_fd_params.append(dae_iter_nums_at.get())

    cur_generate_params.append(seg_length.get())
    cur_generate_params.append(step.get())

    last_stable_params = cur_params.copy()

    th = threading.Thread(target=roll_polling)
    th.setDaemon(True)
    th.start()
    tk.mainloop()


if __name__ == '__main__':
    main()
