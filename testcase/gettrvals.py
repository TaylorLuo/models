
def read_imagefile_label(imagefile_label):

  f = open(imagefile_label)
  f2 = open('./trainval.txt', 'w', encoding='utf8')
  for line in f:
    line = line.split('\t')
    im = line[-1]
    # f2.writeline(im.split('.')[0]+'\n')
    # print(im.split('.')[0]+"'")
    if('02977' in ['02977', '03328', '05223', '05499', '06888', '00470', '06369']):
      print('yes, it is in')


read_imagefile_label('./cars_train_annos.txt')

