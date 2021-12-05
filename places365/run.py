from torch.nn.functional import fractional_max_pool2d
from similar_words import similar
from visualize import display_pca_scatterplot
from PIL import Image
from gensim.models import KeyedVectors
import numpy as np

import os

if __name__ == "__main__":

    moods = ['Happy', 'Funny', 'Chill', 'Dramatic', 'Sad', 'Romantic', 'Serious' , 'Scary' , 'Peaceful']
    video_themes = ['Travel', 'Nature', 'Sports', 'Food', 'City']


    path = "D:\Fall 2021\Deep Learning for Text Data - CSE 8803 DLT\PROJECT\SYNCPHONIC\places365\images\data1"

    scene_info = []
    overall_theme = {}

    for image_path in os.listdir(path):
        input_path = os.path.join(path, image_path)

        #img = Image.open("images/trek.jpeg")
        img = Image.open(input_path)
        img_desc, img_theme = similar(img)

        scene_info.append(img_desc)
        if(img_theme not in overall_theme):
            overall_theme[img_theme] = 1
        else:
            overall_theme[img_theme] += 1
        
        # print(scene_info)
        # print(final_mood)
        # print("_________________")

    scene_info = list(np.concatenate(scene_info).flat)
    
    cleaned_scene_info = []

    for scene in scene_info:
        scene=scene.split('_')[0]
        scene=scene.split('/')[0]
        cleaned_scene_info.append(scene)
    
    print(cleaned_scene_info)
    print(overall_theme)

    final_theme = max(overall_theme, key=overall_theme.get)
    print(final_theme)

    # model = KeyedVectors.load_word2vec_format('D:/Fall 2021/Deep Learning for Text Data - CSE 8803 DLT/PROJECT/SYNCPHONIC/places365/GoogleNews-vectors-negative300-SLIM.bin.gz', binary=True)
    # display_pca_scatterplot(model, cleaned_scene_info, moods, video_themes)

    
    






    

    

