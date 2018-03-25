import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import os
import glob
import urllib2
import caffe

##--------------------------------------------------------------------------------------------------##
# Specifying the model files, generated caffemodel to be used for prediction and mean.binaryproto file
##--------------------------------------------------------------------------------------------------##
MODEL_FILE = '/model/deploy.prototxt'
PRETRAINED = '/model/snapshots__iter_1652.caffemodel'
BINARY_PROTO_MEAN_FILE = "/data/mean.binaryproto"

##-------------------##
# Reading the mean file
##-------------------##
blob = caffe.proto.caffe_pb2.BlobProto()
data = open(BINARY_PROTO_MEAN_FILE, 'rb').read()
blob.ParseFromString(data)
mean_arr = np.array(caffe.io.blobproto_to_array(blob))[0]

##---------------------------##
# Setting up the mode for caffe
##---------------------------##
caffe.set_mode_gpu()
#caffe.set_mode_cpu()

##------------------------------------------------------------------------------------------------------------------------------##
# Specifying the network and its parameters
# @args - 1.Model File
# @args - 2.Trained caffe model
# @args - 3.Mean
# @args - 4.Channel swap
# @args - 5.Raw_scale
# @args - 6. Image dimensions
##-------------------------------------------------------------------------------------------------------------------------------##
net = caffe.Classifier(MODEL_FILE, PRETRAINED, mean=mean_arr.mean(1).mean(1),channel_swap=(2,1,0),raw_scale=255,image_dims=(32,32))


#Counter for processed files
number_of_files_processed=0


##---------------------------------------------------------------------##
# Looping over the test images
##---------------------------------------------------------------------##
for file in glob.glob("/images/*"):
	number_of_files_processed += 1
	FileName = file.split("/")[-1]
	input_image = caffe.io.load_image(file)
	prediction = net.predict([input_image])
	s = FileName+","
	for probability in prediction[0]:
		s+=str(probability)+","
	p = prediction[0].argmax()
	outf.write(FileName+" "+str(p)+"\n")
outf.close()

##-----------------------------------------------------------------##
# Soring the labels in the desired format
##-----------------------------------------------------------------##
f1 = open('/data/Ydata.txt','r')
f2 = open('/data/outputPredict.txt','r')
f3 = open('/output/finalOutput.txt','w')

#c=1
for line in f1:
	for pred in f2:
		if str(line[0:5]) == str(pred[0:5]):
			f3.write(pred)
			break
	f2.seek(0)

##--------------------End of the file------------------------------##
	
