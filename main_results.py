import os
from scipy.io import loadmat
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from ultralytics import YOLO

if __name__ == '__main__':
    mat = loadmat('datasets/test_data.mat')
    test_info = mat['test_info']
    test_files = test_info[0][0][0].flatten()
    data_dir = os.environ['DATA_DIR']
    results = [#'runs/classify/yolov8_SOLUTION_1_classify_s',
               'runs/detect/yolov8_SOLUTION_2_detect_n6']

    for result in results:
        print(result)
        # Load the trained model
        model = YOLO(os.path.join(result, 'weights/best.pt'), verbose=False)
        labels = {breed: indx for indx, breed in model.names.items()}
        y_pred = []
        target_labels = []

        if result.startswith('runs/detect/'):
            output_pred = lambda i: int(i[0].boxes.cls.cpu()[0]) if len(i[0].boxes.cls.cpu())>0 else -1
        elif result.startswith('runs/classify/'):
            output_pred = lambda i: i[0].probs.top1

        for image_fp in test_files:
            target_breed = image_fp[0].split('-', 1)[1].split('/')[0]
            # Predict using trained model
            pred = model.predict(source=os.path.join(data_dir, image_fp[0]),
                                 save=False,
                                 verbose=False)
            y_pred.extend([output_pred(pred)])
            target_labels.extend([labels[target_breed]])

        acc = accuracy_score(target_labels, y_pred)
        print('Accuracy score :', acc)

        # Create a Classification Report
        report = classification_report(target_labels,
                                       y_pred,
                                       labels=None,
                                       digits=2,
                                       output_dict=False)
        print('Classification report :\n', report)
