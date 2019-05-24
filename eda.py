import pandas as pd
import os
import pydicom
import numpy as np
from matplotlib import pyplot as plt

mri_directory = './mri_scans/'

class DatasetAnalyzer:
    def __init__(self, unique_keys, dicom_directory):
        self.keys = unique_keys
        self.root_dir = dicom_directory
        self.scans_list = os.listdir(mri_directory)
        self.organized_folders = {}
    
    def read_dicoms(self):
        for folder in os.listdir(self.root_dir):
            folder = os.path.join(self.root_dir,folder)
            for scan in os.listdir(folder):
                dcm_scan = pydicom.dcmread(os.path.join(folder, scan))
                key_tuple = str((dcm_scan.PatientName,dcm_scan.AccessionNumber))
                if key_tuple not in self.organized_folders.keys():
                    self.organized_folders[key_tuple] = [os.path.join(folder,scan)]
                else:
                    self.organized_folders[key_tuple].append(os.path.join(folder,scan))



dataset_analyzer = DatasetAnalyzer(['PatientName', 'AccessionNumber'],mri_directory)
dataset_analyzer.read_dicoms()
print(dataset_analyzer.organized_folders)