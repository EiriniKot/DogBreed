import os, shutil
from scipy.io import loadmat
from tools import Trainer

if __name__ == '__main__':
    # Load files
    mat = loadmat('datasets/train_data.mat')
    train_info = mat['train_info']
    train_files = train_info[0][0][0].flatten()
    mat = loadmat('datasets/test_data.mat')
    test_info = mat['test_info']
    test_files = test_info[0][0][0].flatten()

    # n = nano is the smaller version of yolov8
    yolo_sizes = ['n']
    apply_solutions = ['SOLUTION_1', 'SOLUTION_2']
    try:
        trainer = Trainer(train_files,
                          test_files,
                          data_directory=os.environ['DATA_DIR'],
                          valid_size=0.2,
                          random_state=42,
                          yolo_sizes=yolo_sizes,
                          model_kwargs={'epochs': 100, #'imgsz': 96,
                                        'batch': 16, 'patience': 25},
                          save=True)
        for solution in apply_solutions:
            print(f'Trying experiment with solution {solution}')
            print('-'*50)
            results = trainer.train(solution)
            print('-'*50)
    except Exception as e:
        print('Exception occured while training')
        print(e)
        shutil.rmtree('datasets/yolov8_imgs')







