import codecs

f = codecs.open('./cars_train_annos_p1——test.txt', "wb", encoding = 'utf-8')
tuple1 = (0.23261910676956177, 0.16466346383094788, 0.8237443566322327, 0.8833049535751343)
labels = 111
pic_name = '1_Label_111.jpg'
recordstr = str(tuple1[0])+'\t'+str(tuple1[1])+'\t'+str(tuple1[2])+'\t'+str(tuple1[3])+'\t'+str(labels)+'\t'+pic_name
f.writelines((recordstr + '\n'))
f.close()



