import os
import argparse
import glob
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
import cv2

def compute_hist(im):
    hist = cv2.calcHist([im], [0, 1, 2], None, [8, 8, 8],
                        [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist).flatten()
    return hist

def match_hists(hists):
    matches = {}
    idx = 0
    for i in range(len(hists)):
        for j in range(len(hists[i])):
            matches['{}_{}'.format(i,j)] = idx
            idx += 1


    for i in range(len(hists)):
        for j in range(i+1, len(hists)):
            for k in range(len(hists[i])):
                corrs = []
                for l in range(len(hists[j])):
                    corr = cv2.compareHist(hists[i][k], hists[j][l], cv2.cv.CV_COMP_INTERSECT)
                    corrs.append(corr)

                corrs = np.array(corrs)
                idxs = np.argsort(corrs)
                if corrs[idxs[-1]] > 0.80:
                    matches['{}_{}'.format(j, idxs[-1])] = matches['{}_{}'.format(i, k)]

    return matches


def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def parse_position(filename):
    class_name = filename.split('/')[-2]
    frame = filename.split('/')[-1].split('_')[1].split('e')[1]
    frame = int(frame)
    cam = filename.split('/')[-1].split('_')[2].split('m')[1]
    cam = cam.split('.')[0]
    cam = int(cam)
    pos_dict = dict(frame=frame,cam=cam, class_name=class_name)
    positions = []
    with open(filename, 'r') as fp:
        for i, line in enumerate(fp.readlines()):
            pos = map(float, line.strip().split())
            new_pos = pos_dict.copy()
            new_pos['x'] = pos[0]
            new_pos['y'] = pos[1]
            new_pos['w'] = pos[2] - pos[0]
            new_pos['h'] = pos[3] - pos[1]
            new_pos['id'] = i+1
            positions.append(new_pos)

    return positions

def add_rect(fig, x, y, w, h, color, name):
    r =  patches.Rectangle((x, y), w, h, facecolor=color, alpha=0.5)
    rx, ry = r.get_xy()
    cx = rx + r.get_width()/2.0
    cy = ry + r.get_height()/2.0

    fig.annotate(name, (cx, cy), color='k', weight='bold',
                fontsize=12, ha='center', va='center')
    fig.add_patch(patches.Rectangle((x, y), w, h, facecolor=color, alpha=0.5))

def plot_positions(df, img_path, frame, cam):
    color_dict = {'car': '#fc8d59', 'bus': '#ffffbf', 'person': '#91cf60'}
    frame_pos = df[(df['frame'] == frame) & (df['cam'] == cam)]
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    im = plt.imread(img_path)
    ax.imshow(im)
    for i, f in frame_pos.iterrows():
        add_rect(ax, f['x'], f['y'], f['w'], f['h'], color=color_dict[f['class_name']], name=f['id'])


    legend_handles = []
    for k, v in color_dict.iteritems():
        handle = patches.Patch(color=v, label=k)
        legend_handles.append(handle)

    plt.legend(loc=0, handles=legend_handles)
    plt.xlim((0, 360))
    plt.ylim((0, 288))
    plt.ylim(plt.ylim()[::-1])
    plt.tight_layout()
    plt.tick_params(axis='both', left='off', top='off', right='off',
                    bottom='off', labelleft='off', labeltop='off',
                    labelright='off', labelbottom='off')
    plt.show()


parser = argparse.ArgumentParser(description='Dataset directory')
parser.add_argument('-d', '--data_dir')

args = parser.parse_args()
classes = ['car', 'bus', 'person']
image_dir = os.path.join(args.data_dir, 'images')
positions = []
for c in classes:
    class_path = os.path.join(args.data_dir, 'positions', c, '*')
    for f in glob.glob(class_path):
        positions.extend(parse_position(f))

df = pd.DataFrame(positions)

cam_images = []
for cam in range(6):
    images = sorted(glob.glob(os.path.join(image_dir, 'c'+str(cam), '*')))
    cam_images.append(images)

# frame = 1
# for cam in range(1, 7):
#     plot_positions(df, cam_images[cam-1][frame-1], frame, cam-1)

for frame in range(1, 242):
    for c in classes:
        hists = []
        for cam in range(6):
            # plot_positions(df, cam, frame)
            img = Image.open(cam_images[cam][frame-1])
            filt_df = df[(df['frame'] == frame)
                         & (df['cam'] == cam)
                         & (df['class_name'] == c)]
            idx = 1
            cam_hists = []
            for i, f in filt_df.iterrows():
                x = int(f['x'])
                y = int(f['y'])
                r = int(f['x'] + f['w'])
                b = int(f['y'] + f['h'])
                if  c == 'person':
                    x,y,r,b = x-3, y-3, r+3, b+3
                cimg = img.crop((x, y, r, b)).resize((32,32), Image.ANTIALIAS)
                directory = args.data_dir + '/out/{}/'.format(frame)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                # cimg.save(directory + '{}_{}_{}_.png'.format(c, cam, idx))
                cimg_np = np.array(cimg)
                hist = compute_hist(cimg_np)
                cam_hists.append(hist)
                idx += 1

            hists.append(cam_hists)
        matches = match_hists(hists)
        assert False
