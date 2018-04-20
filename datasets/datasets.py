import numpy as np
import os
from subprocess import call
from chainer.dataset.download import get_dataset_directory
from chainer.datasets import TupleDataset

def download(url, path):
    call(['wget', url, '-O', path])

def get_mvmc(cam=None, tr_percent=0.5):
    if cam is None:
        cam = np.arange(6)

    url = 'https://www.dropbox.com/s/rofaov8tgqhh6jv/MVMC.npz'
    base_dir = get_dataset_directory('mvmc/')
    path = os.path.join(base_dir, 'mvmc.npz')

    if not os.path.isfile(path):
        download(url, path)

    data = np.load(path)
    X = data['X']
    y = data['y']
    sidx = int(len(y)*tr_percent)
    train = TupleDataset(X[:sidx, cam], y[:sidx, cam])
    test = TupleDataset(X[sidx:, cam], y[sidx:, cam])
    return train, test


def get_mvmc_flatten_eval(cam):
    url = 'https://www.dropbox.com/s/rofaov8tgqhh6jv/MVMC.npz'
    base_dir = get_dataset_directory('mvmc/')
    path = os.path.join(base_dir, 'mvmc.npz')

    if not os.path.isfile(path):
        download(url, path)

    data = np.load(path)
    X = data['X']
    y = data['y']
    
    # Turn 3 to negative -1 for empty view
    y = y.astype(np.int32)
    y[y==3] = -1
    
    # Get the max and
    last = np.max(y,1)
    last = last[:,np.newaxis]
    y = np.hstack([y,last])
    
    ridx = [770, 723, 240, 21, 548, 440, 378, 192, 435, 792, 248, 784, 608, 676, 406, 353, 515, 709, 692, 303, 58, 565, 549, 82, 418, 825, 108, 562, 333, 226, 427, 431, 483, 165, 72, 386, 290, 186, 714, 740, 682, 218, 701, 417, 652, 352, 775, 60, 150, 404, 554, 823, 755, 232, 831, 221, 839, 167, 198, 567, 337, 238, 420, 400, 79, 242, 53, 474, 383, 684, 747, 537, 590, 389, 700, 423, 665, 377, 185, 301, 791, 434, 468, 231, 486, 820, 822, 798, 4, 403, 455, 233, 320, 817, 5, 407, 91, 56, 104, 151, 125, 415, 574, 316, 659, 387, 512, 661, 669, 155, 824, 518, 126, 587, 499, 205, 842, 725, 522, 342, 645, 612, 365, 65, 813, 399, 818, 38, 762, 644, 563, 463, 462, 350, 131, 343, 767, 370, 366, 630, 154, 675, 172, 270, 410, 175, 541, 478, 696, 295, 598, 766, 95, 306, 275, 286, 788, 105, 112, 210, 761, 207, 40, 48, 703, 450, 330, 493, 837, 245, 349, 732, 236, 182, 92, 201, 419, 90, 552, 19, 519, 672, 650, 662, 806, 272, 787, 73, 582, 132, 146, 100, 695, 603, 632, 76, 359, 251, 721, 102, 41, 239, 10, 393, 197, 89, 814, 664, 170, 558, 358, 163, 14, 843, 800, 797, 174, 346, 128, 203, 573, 259, 157, 261, 628, 6, 739, 241, 327, 553, 319, 835, 707, 188, 671, 534, 533, 602, 311, 785, 422, 712, 305, 528, 3, 466, 372, 827, 274, 318, 380, 145, 467, 647, 768, 144, 497, 196, 169, 481, 453, 447, 655, 635, 556, 249, 627, 752, 706, 193, 42, 836, 17, 140, 2, 413, 611, 428, 69, 288, 439, 815, 369, 348, 505, 677, 509, 718, 408, 591, 149, 200, 228, 795, 593, 566, 599, 116, 506, 215, 491, 502, 61, 500, 847, 651, 779, 620, 88, 622, 624, 414, 495, 34, 848, 487, 432, 595, 807, 78, 680, 545, 28, 759, 490, 294, 148, 62, 617, 656, 379, 489, 122, 597, 529, 778, 601, 688, 179, 543, 234, 322, 536, 171, 362, 840, 658, 763, 213, 583, 781, 260, 120, 492, 250, 516, 633, 336, 520, 32, 302, 660, 195, 30, 280, 194, 623, 217, 613, 621, 829, 314, 526, 335, 219, 461, 216, 638, 298, 782, 720, 646, 341, 152, 679, 9, 804, 25, 16, 609, 351, 331, 285, 284, 572, 446, 64, 310, 223, 173, 356, 426, 776, 367, 212, 224, 535, 398, 97, 396, 501, 81, 777, 717, 482, 594, 743, 550, 730, 523, 634, 110, 225, 266, 513, 291, 525, 130, 252, 328, 496, 542, 262, 115, 657, 87, 510, 846, 124, 111, 734, 774, 514, 488, 164, 540, 67, 683, 276, 312, 264, 12, 790, 809, 687, 576, 460, 208, 227, 786, 214, 689, 530, 394, 547, 237, 575, 158, 793, 589, 304, 765, 103, 637, 799, 833, 267, 796, 329, 22, 674, 570, 202, 607, 273, 719, 726, 639, 850, 409, 555, 246, 812, 849, 143, 18, 209, 39, 698, 577, 475, 255, 636, 15, 364, 485, 448, 473, 412, 697, 20, 728, 438, 578, 52, 129, 405, 610, 760, 470, 600, 268, 702, 35, 371, 421, 769, 168, 55, 653, 773, 7, 161, 810, 693, 166, 744, 385, 181, 464, 334, 616, 605, 24, 517, 841, 147, 59, 504, 524, 465, 243, 751, 457, 156, 71, 816, 74, 564, 772, 83, 265, 789, 724, 731, 384, 134, 640, 1, 584, 568, 592, 569, 381, 68, 844, 561, 794, 220, 402, 629, 33, 136, 299, 783, 98, 139, 47, 430, 325, 309, 199, 614, 27, 293, 531, 451, 459, 749, 507, 44, 388, 764, 802, 46, 176, 416, 93, 673, 382, 70, 729, 424, 803, 77, 159, 663, 292, 711, 780, 588, 355, 436, 753, 94, 184, 141, 667, 375, 705, 832, 49, 626, 138, 756, 750, 737, 449, 425, 50, 80, 229, 123, 397, 106, 75, 376, 162, 137, 472, 296, 654, 694, 585, 354, 8, 811, 178, 643, 307, 317, 571, 315, 494, 269, 666, 187, 37, 704, 230, 452, 107, 222, 191, 579, 411, 287, 819, 648, 36, 771, 357, 443, 433, 521, 0, 681, 742, 401, 118, 360, 503, 13, 339, 189, 297, 722, 374, 31, 715, 135, 277, 758, 469, 757, 685, 395, 326, 670, 532, 690, 508, 109, 801, 99, 631, 142, 281, 43, 256, 838, 258, 373, 544, 313, 347, 713, 476, 527, 604, 283, 686, 480, 539, 429, 845, 581, 538, 153, 121, 253, 63, 748, 727, 235, 160, 247, 23, 477, 278, 641, 668, 66, 586, 323, 279, 805, 363, 437, 86, 391, 444, 180, 117, 557, 691, 625, 615, 289, 190, 821, 254, 546, 808, 11, 442, 204, 738, 211, 699, 282, 826, 456, 471, 26, 551, 361, 96, 710, 735, 271, 57, 458, 29, 332, 324, 338, 716, 114, 177, 619, 741, 308, 119, 618, 642, 830, 834, 445, 345, 733, 580, 560, 479, 828, 484, 606, 441, 708, 511, 113, 498, 51, 101, 45, 340, 85, 454, 390, 649, 754, 745, 392, 133, 596, 559, 244, 746, 321, 127, 678, 206, 263, 300, 257, 368, 84, 344, 54, 183, 736]
    tr_percent = 0.8
    
    sidx = int(len(X)*tr_percent)
    
    Xtrain = X[ridx[:sidx]][:,cam]
    ytrain = y[ridx[:sidx]][:,cam+[6]]
    
    Xtest = X[ridx[sidx:]][:,cam]
    ytest = y[ridx[sidx:]][:,cam+[6]]
        
    train_xs = Xtrain.transpose((1,0,2,3,4)).tolist()
    train_xs = [np.array(train_x).astype(np.float32) for train_x in train_xs]
    train_ys = ytrain.transpose((1,0)).tolist()
    train_ys = [np.array(train_y).astype(np.int32) for train_y in train_ys]
    
    test_xs = Xtest.transpose((1,0,2,3,4)).tolist()
    test_xs = [np.array(test_x).astype(np.float32) for test_x in test_xs]
    test_ys = ytest.transpose((1,0)).tolist()
    test_ys = [np.array(test_y).astype(np.int32) for test_y in test_ys]
    
    train = TupleDataset(*(train_xs + train_ys))
    test = TupleDataset(*(test_xs + test_ys))
    
    #train = permute(train)
    #test = permute(test)
    
    return train, test

def rotate(l, n):
    return l[n:] + l[:n]

def permute(train):
    train_xs = []
    train_ys = []
    for t in train:
        l = len(t)
        x = t[:l//2]
        y = t[l//2:]
        train_xs.append(x)
        train_ys.append(y)
        for i in range(5):
            x = rotate(x,1)
            y = rotate(y,1)
            train_xs.append(x)
            train_ys.append(y)
        
    train_xs = np.array(train_xs).transpose((1,0,2,3,4))
    train_xs = [x for x in train_xs]
    train_ys = np.array(train_ys).transpose((1,0))
    train_ys = [y for y in train_ys]
    #print(train_xs.shape)
    #print(train_ys.shape)
    return TupleDataset(*(train_xs + train_ys))

def get_mvmc_flatten(cam=None, tr_percent=0.5):
    if cam is None:
        cam = np.arange(6).tolist()

    url = 'https://www.dropbox.com/s/uk8c6iymy8nprc0/MVMC.npz'
    base_dir = get_dataset_directory('mvmc/')
    path = os.path.join(base_dir, 'mvmc.npz')

    if not os.path.isfile(path):
        download(url, path)

    data = np.load(path)
    X = data['X']
    y = data['y']
    
    # Turn 3 to negative -1 for empty view
    y = y.astype(np.int32)
    y[y==3] = -1
    
    # Get the max and
    last = np.max(y,1)
    last = last[:,np.newaxis]
    y = np.hstack([y,last])
    
    ridx = np.random.permutation(range(len(X))).tolist()
    
    sidx = int(len(X)*tr_percent)
    
    Xtrain = X[ridx[:sidx]][:,cam]
    ytrain = y[ridx[:sidx]][:,cam]
    
    Xtest = X[ridx[sidx:]][:,cam]
    ytest = y[ridx[sidx:]][:,cam]
        
    train_xs = Xtrain.transpose((1,0,2,3,4)).tolist()
    train_xs = [np.array(train_x).astype(np.float32) for train_x in train_xs]
    train_ys = ytrain.transpose((1,0)).tolist()
    train_ys = [np.array(train_y).astype(np.int32) for train_y in train_ys]
    
    test_xs = Xtest.transpose((1,0,2,3,4)).tolist()
    test_xs = [np.array(test_x).astype(np.float32) for test_x in test_xs]
    test_ys = ytest.transpose((1,0)).tolist()
    test_ys = [np.array(test_y).astype(np.int32) for test_y in test_ys]
        
    train = TupleDataset(*(train_xs + train_ys))
    test = TupleDataset(*(test_xs + test_ys))
    
    train = permute(train)
    test = permute(test)
    
    return train, test
