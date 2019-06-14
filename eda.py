import pandas as pd
import os
import pydicom
import numpy as np
from matplotlib import pyplot as plt

mri_directory = './mri_data/linking-anna-alex-abhi-2/'


class DatasetAnalyzer:
    def __init__(self, unique_keys, dicom_directory):
        self.keys = unique_keys
        self.root_dir = dicom_directory
        self.scans_list = os.listdir(mri_directory)
        self.organized_folders = {}
        self.dcm_dict = {}

    def read_dicoms(self):

        for folder in os.listdir(self.root_dir):
            folder = os.path.join(self.root_dir, folder)
            for scan in os.listdir(folder):
                dcm_scan = pydicom.dcmread(os.path.join(folder, scan))
                dcm_keys = dcm_scan.dir("")
                for key in dcm_keys:
                    value = str(dcm_scan.data_element(key)._value)
                    tag = str(dcm_scan.data_element(key).tag)
                    if tag not in self.dcm_dict:
                        self.dcm_dict[tag] = {}
                    if value not in self.dcm_dict[tag]:
                        self.dcm_dict[tag][value] = 0
                    self.dcm_dict[tag][value] += 1

    def quot(self, string):
        return "\"" + str(string) + "\""

    def write_dcm_dict(self, fn):
        # dcm_keys = dcm_scan.dir("")
        # print(dcm_keys)
        # key_tuple = str((dcm_scan.PatientName,dcm_scan.AccessionNumber))
        # if key_tuple not in self.organized_folders.keys():
        # 	self.organized_folders[key_tuple] = [os.path.join(folder,scan)]
        # else:
        # 	self.organized_folders[key_tuple].append(os.path.join(folder,scan))
        f = open(fn, "w")

        # sort the lists
        value_count_list = []

        for tag, values in self.dcm_dict.items():

            # (value, count) tuples
            count_list = sorted(values.items(), key=lambda x: x[1])
            value_count_list.append(count_list)

        value_count_list = sorted(value_count_list, key=lambda())


        # write the sorted lists
        for tag, values in value_count_list:
            for value, count in values.items():
                if tag != "(7fe0, 0010)":
                    f.write(self.quot(tag) + "," + self.quot(value) + "," + self.quot(count) + "\n")


        f.close()


dataset_analyzer = DatasetAnalyzer(['PatientName', 'AccessionNumber'], mri_directory)
dataset_analyzer.read_dicoms()
dataset_analyzer.write_dcm_dict("dicom_element_counts.csv")
# print(dataset_analyzer.organized_folders)
