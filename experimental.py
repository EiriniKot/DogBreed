import scipy.io
import os
from tools import convert_annotation

mat = scipy.io.loadmat('datasets/test_data.mat')
test_data = mat['test_data']
test_fg_data = mat['test_fg_data']
print(F"The test datasets have shape: {test_data.shape}")
print(F"Test samples are :{test_data.shape[0]}")
test_info = mat['test_info']
files = test_info[0][0][0].flatten()
tst_annotation = test_info[0][0][1].flatten()
assert len(files) == len(tst_annotation)
labels = test_info[0][0][2]

###

cwd = os.getcwd()
output_path = 'datasets/yolov8_labels'
os.makedirs(output_path, exist_ok=True)
classes = [lbl.split("-", 1)[1] for lbl in os.listdir("datasets/Dataset/Images")]

for dir_path in tst_annotation:
    full_dir_path = os.path.join("datasets/Annotation", dir_path[0])
    convert_annotation(full_dir_path, output_path, classes)
    print("Finished processing: " + dir_path[0])

# print(F"The test info has length: {len(test_info[0][0][0])}")
# # print(F"test info labels {test_info[0][0][2]}")
# print(F"The number of classes are :{len(np.unique(np.array(test_info[0][0][2])))}")
#
# mat = scipy.io.loadmat('datasets/file_list.mat')
