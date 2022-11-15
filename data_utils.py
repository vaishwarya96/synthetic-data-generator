#!/usr/bin/env python
# coding: utf-8


import os
import numpy as np
import cv2
import csv
import pickle
import torch



IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def extract(path, ext, exception="/") : 
    paths = []
    name_dir = []
    for root, dirs, files in os.walk(path):
        for name in dirs:
            if name.endswith(ext) and not name.startswith(exception):        
                paths.append(os.path.join(root, name))
                name_dir.append(dirs)
        for name in files:
            if name.endswith(ext) and not name.startswith(exception) :        
                paths.append(os.path.join(root, name))
                name_dir.append(os.path.basename(root))
    filenames = [ os.path.splitext( os.path.basename(l))[0]  for l in paths]
    basenames = [  "".join(os.path.splitext( os.path.basename(l)))  for l in paths]

    sort_dir = sorted(set(name_dir))

    return paths, basenames, filenames, {sort_dir[i]: i for i in range(len(sort_dir)) }

def check_dirs(filename):
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    
def get_selected_taxons(file_path, inv=False):
    selected_taxons = {}
    f = open(file_path, 'r') 
    lines = f.readlines() 
    del lines[0]
    for line in lines:
        line = line.strip()
        taxon, taxon_id = line.split(',')[0], int(line.split(',')[1])
        selected_taxons[taxon] = taxon_id
    if inv:
        return {v: k for k, v in selected_taxons.items()} 
    else:
        return selected_taxons
    
def delete_all_files_in_folder(folder):
    print("Deleting all files in", folder)
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
            
def return_all_files_in_folder_rec(paths, exts=None):
    if not isinstance(paths, list):
        paths = [paths]
    images = []
    for path in paths:
        for (dirpath, dirnames, filenames) in os.walk(path):
            for filename in filenames:
                if exts is None or filename.split(".")[-1] in exts:
                    images.append(os.path.join(path,filename))
                else:
                    print(filename, "rejected!")
    return images

def array_to_csv(array, file, delimiter=","):
    check_dirs(file)
    print("Creating:",file)
    f = open(file, 'w')
    with f:
        writer = csv.writer(f, delimiter=delimiter)
        for row in array:
            if not isinstance(row, list):
                row = [row]
            writer.writerow(row)
            
def save_img(img, path, compress=0):
    check_dirs(path)
    if cv2.imwrite(path,img,  [cv2.IMWRITE_PNG_COMPRESSION, compress]):
        print("Image", path, "saved sucessfully!")
    else:
        print("Error while saving",path)
    
def savePickle(obj, path):
    check_dirs(path)
    pickle_out = open(path,"wb")
    pickle.dump(obj, pickle_out)
    pickle_out.close()
    
def add_path(path):
    # add path to python path
    if path not in sys.path:
        sys.path.insert(0, path)
        
def read_csv(path):
    tmp=[]
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
            tmp.append(row[0])
    return tmp
def balance_dataset(X_train, y_train, max_samples=None):
    train_dict = {}
    for file, label in zip(X_train, y_train):
        train_dict.setdefault(label, []).append(file)
    if max_samples is None: max_samples = np.max([len(train_dict[taxon_id]) for taxon_id in train_dict])
    X_train = []
    y_train = []
    for taxon_id in train_dict:
        ratio = np.ceil(max_samples/len(train_dict[taxon_id]))
        tmp = np.repeat(train_dict[taxon_id], ratio)
        np.random.shuffle(tmp)
        train_dict[taxon_id] = tmp[0:max_samples]
        X_train.extend(tmp[0:max_samples])
        y_train.extend([taxon_id]*max_samples)
    print("Balanced to %d samples per class!" %(max_samples))
    return X_train, y_train, max_samples

def load_checkpoint(filepath, device):

    # checkpoint = torch.load(filepath)

    # model = checkpoint['model']
    # model.load_state_dict(checkpoint['state_dict'])
    # for parameter in model.parameters():
    #     parameter.requires_grad = False


    model = torch.jit.load(".".join(filepath.split('.')[:-1] ) + ".tjm")
    model = model.to(device)
    model.eval()
    return model

# %%
