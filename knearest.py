# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 07:19:14 2018

@author: Diko
"""
from csv import reader
from sys import exit
from math import sqrt
from operator import itemgetter
import numpy as np
import csv
import os

def load_data_set(filename): #load file csv
    try:
        with open(filename, newline='') as iris:
            return list(reader(iris, delimiter=','))
    except FileNotFoundError as e:
        raise e
        
def convert_data(data,filename): #Hapus index,header pada datatugas
    data_file = open(filename, 'wt', newline ='')
    with data_file:
        writer = csv.writer(data_file, delimiter=',')
        writer.writerows(data)
        
def convert_to_float(data_set, mode): #masukkan csv ke array
    new_set = []
    if mode == 'training':
        for data in data_set:
            new_set.append([float(x) for x in data[:len(data)-1]] + [data[len(data)-1]])
    elif mode == 'test':
        for data in data_set:
            new_set.append([float(x) for x in data])
    else:
        print('Invalid mode, program will exit.')
        exit()
    return new_set

def get_classes(training_set): #Mendapatkan nilai Y pada datatrain
    return list(set([c[-1] for c in training_set]))

def find_neighbors(distances, k): #mendapatkan neighbor 
    return distances[0:k] 

def find_response(neighbors, classes):
    votes = [0] * len(classes)
    for instance in neighbors:
        for ctr, c in enumerate(classes):
            if instance[-2] == c:
                votes[ctr] += 1
    return max(enumerate(votes), key=itemgetter(1))

def knn(training_set, test_set, k):
    distances = [] #array untuk menampung nilai jarak
    temp = [] #array untuk menampung output
    dist = 0 #variabel jarak
    limit = len(training_set[0]) - 1 #mencari nilai Y
    # generate response classes from training data
    classes = get_classes(training_set)
    for test_instance in test_set: 
        for row in training_set:
            for x, y in zip(row[:limit], test_instance):
                dist += (x-y) * (x-y)     #loop untuk mencari nilai jarak          
            distances.append(row + [sqrt(dist)])
            dist = 0    
        distances.sort(key=itemgetter(len(distances[0])-1))
        # find k nearest neighbors
        neighbors = find_neighbors(distances, k)
        # get the class with maximum votes
        index, value = find_response(neighbors, classes)
        temp.append([str(test_instance),classes[index]]) #menampung output ke array yg telah dibuat
        distances.clear()

    with open('TebakanTugas3.csv', 'w+') as f: #write array ke file
        for item in temp:
            f.write("%s\n" % item)
    print(temp)
        
#mengambil data dari file csv dgn lib dari numpy
data_train = np.genfromtxt('DataTrain_Tugas3_AI.csv', delimiter=',', skip_header=1) 
data_test  = np.genfromtxt('DataTest_Tugas3_AI.csv', delimiter=',', skip_header=1)

#memilih column yang ingin diambil
indexRemovedTrain = data_train[:,1:7] #column ke 1 hingga 7 dengan semua row
indexRemovedTest  = data_test[:,1:6] #column ke 1 hingga 6 dengan semua row

#mengubah data ke file csv sementara
convert_data(indexRemovedTrain,'convertedtrain.csv') #mengubah data 
convert_data(indexRemovedTest,'convertedtest.csv')

k = int(input('Enter the value of k : ')) #user menentukan nilai K

# load the training and test data set
#ubah file csv sementara ke dalam bentuk array
training_set = convert_to_float(load_data_set('convertedtrain.csv'), 'training') 
test_set = convert_to_float(load_data_set('convertedtest.csv'), 'test')
if k > len(training_set):
    print('Jumlah K yang ditentukan melebihi data training')
else:
    knn(training_set, test_set, k) #fungsi utama program
#hapus file temporary
os.remove("convertedtrain.csv")
os.remove("convertedtest.csv")


