# =============================================================================
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Routine for decoding the NUS-WIDE binary file format."""

import os
import cv2
import numpy as np
import random

# Process images of this size. Note that this differs from the original nus-wide
# image size of 224 x 224. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.

# Global constants describing the NUS-WIDE data set.
class Kinetics(object):
    def __init__(self, modal, data_root, path, train=True):
        self.lines = open(path, 'r').readlines()
        self.data_root = data_root
        self.n_samples = len(self.lines) # num of samples (see lines in train.csv)
        self.n_GOP = 12 # or smaller: 3
        self.train = train
        assert modal == 'video'
        self.modal = 'video'
        self._keyframe = [0] * self.n_samples
        self._pframe = [0] * self.n_GOP
        self._numframe = 0
        self._load = [0] * self.n_samples
        self._load_num = 0
        self._status = 0
        self.data = self.video_data
        self.all_data = self.video_all_data

    def get_video(self, line):
        '''loading video clip '''
        path = os.path.join(self.data_root, self.line[line].strip().split()[0])
        FPS = 30
        buff = []
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_FPS, FPS) # try: the first n_GOP frames
        for n_gop in range(self.n_GOP): # 
            while(cap.isOpended()):
                ret, np_frame = cap.read()
                if ret == True:
                    np_frame = cv2.resize(np_frame, (256+28, 256+28)) # during training randoming select 224x224 crops
                    #np_frame  = cv2.resize(cv2.imread('video', frame), (224, 224)) # resize for speed
                    buff.append(np_frame)
                else:
                    break
        cap.release()
        cv2.destroyAllWindows()
        print("buff shape", buff.shape())
        return buff # (num_frame, 256, 256, 3)

    #def get_file_label(self, line):
    #    '''weak label info from filename '''
    #    return self._data_root.strip().split()[-1]
    
    def get_label(self, i): #len(label)==10
        return [int(j) for j in self.lines[i].strip().split()[1:]]

    def video_data(self, indexes): # split by GOP size == 12
        if self._status:
            return (self._img[indexes,:], self._label[indexes, :])
        else:
            buff_img = []
            buff_label = []
            for i in indexes:
                try:
                    if self.train:
                        if not self._load[i]:
                            for j in self.n_GOP:
                                self._img[j] = self.get_video(i)[(j*self.n_GOP):j*(self.n_GOP+1)]
                                self._label[j] = self.get_label(i)
                                buff_img.append(self._img[j])
                                buff_label.append(self._label[j])
                            self._load[i] = 1
                            self._load_num += 1
                except Exception as e:
                    print('cannot open {}, exception: {}'.format(self.lines[i].strip(), e))
            if self._load_num == self.n_samples:
                self._status = 1
                self._img = np.asarray(self._img)
                self._label = np.asarray(self._label)
            return (np.asarray(buff_img), buff_label)

    def video_all_data(self):
        if self._status:
            return (self._img, self._label)

    def get_file_labels(self):
        '''convert to numpy '''
        for i in range(self.n_sample):
            if self._label[i] is not list:
                self._label[i] = self.lines[i].strip().split()[1:] 
        # return np.asarray(self._label)
        return self._label

def import_train_vid(data_root, vid_tr):
    '''return (vid_tr, txt_tr) '''
    return (Kinetics('video', data_root, vid_tr, train=True))

def import_val_vid(data_root, vid_te, vid_db):
    '''return (vid_te, txt_te, vid_db, txt_db)'''
    print('*** check ***')
    return (Kinetics('video', data_root, vid_te, train=False),
            Kinetics('video', data_root, vid_db, train=False))

class Dataset(object):
    def __init__(self, modal, data_root, path, train=True):
        self.lines = open(path, 'r').readlines()
        self.data_root = data_root
        self.n_samples = len(self.lines)
        self.train = train
        assert modal == 'img'
        self.modal = 'img'
        self._img = [0] * self.n_samples
        self._label = [0] * self.n_samples
        self._load = [0] * self.n_samples
        self._load_num = 0
        self._status = 0
        self.data = self.img_data
        self.all_data = self.img_all_data

    def get_img(self, i):
        path = os.path.join(self.data_root, self.lines[i].strip().split()[0])
        return cv2.resize(cv2.imread(path), (256, 256))

    def get_label(self, i):
        return [int(j) for j in self.lines[i].strip().split()[1:]]

    def img_data(self, indexes):
        '''shape(n_samples, num_frames, 256, 256, 3) '''
        if self._status:
            return (self._img[indexes, :], self._label[indexes, :])
        else:
            ret_img = []
            ret_label = []
            for i in indexes:
                try:
                    if self.train:
                        if not self._load[i]:
                            self._img[i] = self.get_img(i)
                            self._label[i] = self.get_label(i)
                            self._load[i] = 1
                            self._load_num += 1
                        ret_img.append(self._img[i])
                        ret_label.append(self._label[i])
                    else:
                        self._label[i] = self.get_label(i)
                        ret_img.append(self.get_img(i))
                        ret_label.append(self._label[i])
                except Exception as e:
                    print('cannot open {}, exception: {}'.format(self.lines[i].strip(), e))

            if self._load_num == self.n_samples:
                self._status = 1
                self._img = np.asarray(self._img)
                self._label = np.asarray(self._label)
            return (np.asarray(ret_img), np.asarray(ret_label))

    def img_all_data(self):
        if self._status:
            return (self._img, self._label)

    def get_labels(self):
        for i in range(self.n_samples):
            if self._label[i] is not list:
                self._label[i] = [int(j)
                                  for j in self.lines[i].strip().split()[1:]]
        return np.asarray(self._label)


def import_train(data_root, img_tr):
    '''
    return (img_tr, txt_tr)
    '''
    return (Dataset('img', data_root, img_tr, train=True))


def import_validation(data_root, img_te, img_db):
    '''
    return (img_te, txt_te, img_db, txt_db)
    '''
    return (Dataset('img', data_root, img_te, train=False),
            Dataset('img', data_root, img_db, train=False))
