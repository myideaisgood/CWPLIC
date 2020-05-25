import os
import sys
import cv2
import numpy as np

from config import *


def make_buffer(path, filelist, ctx_up, ctx_left, ctx_total):
    num_data = 0

    for idx in range(len(filelist)):
        filename = path + filelist[idx]
        sys.stdout.flush()

        img = cv2.imread(filename)
        height, width, _ = np.shape(img)

        num_data += (height - ctx_up) * (width - ctx_left * 2)

    Xdata = np.arange(0, num_data * (3 * ctx_total -3)).reshape(num_data, (3 * ctx_total - 3))
    Ydata = np.arange(0, 3 * num_data).reshape(num_data, 3)

    return Xdata, Ydata

def rgb2yuv(img):

    r, g, b = cv2.split(img)

    r = np.asarray(r, float)
    g = np.asarray(g, float)
    b = np.asarray(b, float)

    u = b - np.round((87 * r + 169 * g) / 256.0)
    v = r - g
    y = g + np.round((86 * v + 29 * u) / 256.0)

    return y, u, v

def create_dataset(args, data_type):

    path = args.data_dir + data_type + "/"
    ctx_up = args.ctx_up
    ctx_left = args.ctx_left
    ctx_total = (ctx_left * 2 + 1) * ctx_up + ctx_left

    # Read in files
    filelist = os.listdir(path)
    filelist.sort()

    # Make empty buffer for Xdata, Ydata
    Xdata, Ydata = make_buffer(path, filelist, ctx_up, ctx_left, ctx_total)

    print('num_file = ' + str(len(filelist)))

    # Make dataset
    n = 0

    for idx in range(len(filelist)):
        filename = path + filelist[idx]
        print("Reading " + filename)
        sys.stdout.flush()

        img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)

        img_y, img_u, img_v = rgb2yuv(img)

        height, width = np.shape(img_y)

        for y in range(height):
            if y >= ctx_up:
                for x in range(width):
                    if ctx_left <= x < width - ctx_left:
                        v_y = img_y[y - ctx_up:y + 1, x - ctx_left:x + ctx_left + 1].reshape(1, (ctx_left * 2 + 1) * (ctx_up + 1))
                        v_u = img_u[y - ctx_up:y + 1, x - ctx_left:x + ctx_left + 1].reshape(1, (ctx_left * 2 + 1) * (ctx_up + 1))
                        v_v = img_v[y - ctx_up:y + 1, x - ctx_left:x + ctx_left + 1].reshape(1, (ctx_left * 2 + 1) * (ctx_up + 1))

                        ref_left_y = int(v_y[0, ctx_total - 1])
                        ref_left_u = int(v_u[0, ctx_total - 1])
                        ref_left_v = int(v_v[0, ctx_total - 1])

                        Ydata[n, 0] = int(v_y[0, ctx_total]) - ref_left_y
                        Ydata[n, 1] = int(v_u[0, ctx_total]) - ref_left_u
                        Ydata[n, 2] = int(v_v[0, ctx_total]) - ref_left_v

                        for j in range(ctx_total - 1):
                            Xdata[n, j] = int(v_y[0, j]) - ref_left_y
                            Xdata[n, j + ctx_total - 1] = int(v_u[0, j]) - ref_left_u
                            Xdata[n, j + 2 * ctx_total - 2] = int(v_v[0, j]) - ref_left_v

                        n = n + 1

    filename_x = args.data_dir + 'npy/Xdata_' + data_type + '.npy'

    filename_y = args.data_dir + 'npy/Ydata_' + data_type + '.npy'

    np.save(filename_x, Xdata)
    np.save(filename_y, Ydata)

if __name__ == "__main__":

    args = parse_args()
    create_dataset(args, "test")