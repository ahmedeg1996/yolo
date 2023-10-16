import subprocess
from multiprocessing import Pool
import os

def run_prediction(image_path):
    command = f"python segment/predict.py --retina-masks --weights segment/big_model_2nd_batch.pt --conf 0.9 --source {image_path} --save-txt"
    subprocess.run(command, shell=True)

if __name__ == '__main__':
    image_folder = 'C:/Users/v23ASayed2/Desktop/Vodafone/National_IDs_splitting/images'
    image_files = [os.path.join(image_folder, img) for img in os.listdir(image_folder)]
    
    with Pool(processes=2) as pool:
        pool.map(run_prediction, image_files)
