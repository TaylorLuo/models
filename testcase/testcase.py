import chardet

#!/usr/bin/python
# coding: utf-8


def read_imagefile_label(imagefile_label):
  f = open(imagefile_label)
  imagefiles = []
  labels = []
  bboxs = []
  for line in f:
    line = line.split('\t')
    im = line[-1]
    l = line[-2]
    bbox = line[0:4]
    for idx, value in enumerate(bbox):
      bbox[idx] = int(bbox[idx])
    imagefiles.append(im)
    labels.append(int(l))
    bboxs.append(bbox)
  dict = {'filename': imagefiles, 'class': labels, 'bboxs': bboxs}
  return dict


read_imagefile_label('./train_annos.txt')

# line = "39	116	569	375	14	'00001.jpg'"
# line = "30	52	246	147	'00001.jpg'"
# line = line.split('\t')
# im = line[-1]
# l = line[-2]
# bbox = line[0:4]
# for idx, value in enumerate(bbox):
#   bbox[idx] = int(bbox[idx])
# print(im)
# print(int(l))
# print(bbox)