import numpy as np
def label2onehot(element):
    if element == '1':
        return [1,0,0,0,0]
    elif element == '2':
        return [0,1,0,0,0]
    elif element == '3':
        return [0,0,1,0,0]
    elif element == '4':
        return [0,0,0,1,0]
    elif element == '5':
        return [0,0,0,0,1]

def Return_train():
  file = open("ECG5000_TRAIN")
  lines = file.readlines()
  alldata = []
  for line in lines:
    line = line.split(',')
    label = line[0]
    label = np.array(label2onehot(label))
    train = line[1:]
    train = np.array([float(i) for i in train])
    seqlen = np.array(len(train))
    alldata.append((train,label,seqlen))
    #sys.stdout.flush()
  #for i in alldata:
  #    print(i[0])
  return alldata

def Return_test():
  file = open("ECG5000_TEST")
  lines = file.readlines()
  alldata = []
  for line in lines:
    line = line.split(',')
    label = line[0]
    label = np.array(label2onehot(label))
    train = line[1:]
    train = np.array([float(i) for i in train])
    seqlen = np.array(len(train))
    alldata.append((train,label,seqlen))
    #sys.stdout.flush()
  #for i in alldata:
  #    print(i[0])
  return alldata