import os
import glob
import argparse
import numpy as np
import copy

from PIL import Image

def details(filename):
    class_name = filename.split('/')[-1].split('_')[0]
    cam = filename.split('/')[-1].split('_')[1]
    cam = int(cam)
    oid = filename.split('/')[-1].split('_')[2].split('.')[0]
    oid = int(oid)
    return {'class': class_name, 'cam': cam, 'oid': oid, 'path': filename}


parser = argparse.ArgumentParser(description='Dataset directory')
parser.add_argument('-d', '--data_dir')

args = parser.parse_args()
classes = ['car', 'bus', 'person']
image_dir = os.path.join(args.data_dir, 'annotated')

xs, ys = [], []
for i in range(1, 242):
    root = os.path.join(image_dir, str(i))
    imgs = glob.glob(os.path.join(root, '*'))
    if len(imgs) == 0:
        continue

    dets = []
    for img in imgs:
        try:
            det = details(img)
        except:
            continue

        if det['path'].split('.')[0][-1] == '_':
            continue
        dets.append(details(img))

    res = []
    while(len(dets) > 0):
        x, y = [], []
        root = copy.copy(dets[0])
        for cam in range(6):
            idxs = []
            sample = []
            cam_found = False
            for i, det in enumerate(dets):
                if det['cam'] == cam and det['oid'] == root['oid'] and det['class'] == root['class']:
                    sample.append(det)
                    im = np.array(Image.open(det['path'])).astype(np.float32)
                    im = np.rollaxis(im, 2, 0)
                    im /= im.max()
                    x.append(im)
                    y.append(classes.index(det['class']))
                    idxs.append(i)
                    cam_found = True
            if not cam_found:
                x.append(np.zeros((3,32,32)))
                y.append(len(classes))

            # print(sample)
            dets = [i for j, i in enumerate(dets) if j not in idxs]

        xs.append(x)
        ys.append(y)

xs = np.array(xs, dtype=np.float32)
ys = np.array(ys, dtype=np.int32)

np.savez_compressed('MVMC.npz', X=xs, y=ys)
