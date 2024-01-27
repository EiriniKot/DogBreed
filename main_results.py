import os
from scipy.io import loadmat
from sklearn.metrics import classification_report, confusion_matrix
from ultralytics import YOLO

if __name__ == '__main__':
    mat = loadmat('datasets/test_data.mat')
    test_info = mat['test_info']
    test_files = test_info[0][0][0].flatten()
    target_labels = test_info[0][0][2].flatten()
    data_dir = os.environ['DATA_DIR']
    results = ['runs/classify/yolov8_SOLUTION_1_classify_n22',
               'runs/detect/yolov8_SOLUTION_2_detect_n9']

    for result in results:
        print(result)
        # Load the trained model
        model = YOLO(os.path.join(result, 'weights/best.pt'), verbose=False)
        y_pred = []

        for image_fp in test_files:
            pred = model.predict(source=os.path.join(data_dir, image_fp[0]),
                                 save=False,
                                 imgsz=32,
                                 verbose=False)
            y_pred.extend([pred[0].probs.top1])

        # Create a Classification Report
        report = classification_report(target_labels,
                                       y_pred,
                                       labels=None,
                                       digits=2,
                                       output_dict=False)
        # Create Confusion Matrix
        cf_mtrx = confusion_matrix(target_labels,
                                   y_pred,
                                   labels=None)
        print(report)
        print('Confusion Matrix:\n', cf_mtrx)
