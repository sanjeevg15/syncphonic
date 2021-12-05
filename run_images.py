from torch.nn.functional import fractional_max_pool2d
from similar_words import similar
from visualize import display_pca_scatterplot
from PIL import Image
from gensim.models import KeyedVectors
import numpy as np

import moviepy.editor as mpe

import os
import cv2
import glob

def add_audio(path, theme, mood):

    video_name ="D:\Fall 2021\Deep Learning for Text Data - CSE 8803 DLT\PROJECT\SYNCPHONIC\places365\images\output_videos\mygeneratedvideo.mp4"
    audio_name = "D:/Fall 2021/Deep Learning for Text Data - CSE 8803 DLT/PROJECT/SYNCPHONIC/places365/audio/"+final_theme+"/upbeat.mp3"

    my_clip = mpe.VideoFileClip(video_name)
    audio_background = mpe.AudioFileClip(audio_name)
    final_audio = mpe.CompositeAudioClip([audio_background])
    # final_clip = my_clip.set_audio(final_audio)
    my_clip.audio = final_audio

    my_clip.write_videofile("D:/Fall 2021/Deep Learning for Text Data - CSE 8803 DLT/PROJECT/SYNCPHONIC/places365/images/output_videos/new_filename.mp4", fps=25)

    return my_clip




# Video Generating function
def generate_video(path):
    # image_folder = path
    video_name = "D:\Fall 2021\Deep Learning for Text Data - CSE 8803 DLT\PROJECT\SYNCPHONIC\places365\images\output_videos\mygeneratedvideo.mp4"
    os.chdir(path)
      
    images = [img for img in os.listdir(path)
              if img.endswith(".jpg") or
                 img.endswith(".jpeg") or
                 img.endswith("png")]
     
    # Array images should only consider
    # the image files ignoring others if any
    print(images) 
  
    frame = cv2.imread(os.path.join(path, images[0]))
  
    # setting the frame width, height width
    # the width, height of first image
    height, width, layers = frame.shape  
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter(video_name, fourcc, 1, (width, height)) 
  
    # Appending the images to the video one by one
    for image in images: 
        video.write(cv2.imread(os.path.join(path, image))) 
      
    # Deallocating memories taken for window creation
    cv2.destroyAllWindows() 
    video.release()  # releasing the video generated

    return video
  

def video_preprocess(path):
    
    # Folder which contains all the images
    # from which video is to be generated
    os.chdir(path)  

    mean_height = 0
    mean_width = 0
    
    num_of_images = len(os.listdir('.'))
    # print(num_of_images)
    
    for file in os.listdir('.'):
        im = Image.open(os.path.join(path, file))
        width, height = im.size
        mean_width += width
        mean_height += height
        # im.show()   # uncomment this for displaying the image
    
    # Finding the mean height and width of all images.
    # This is required because the video frame needs
    # to be set with same width and height. Otherwise
    # images not equal to that width height will not get 
    # embedded into the video
    mean_width = int(mean_width / num_of_images)
    mean_height = int(mean_height / num_of_images)
    
    # print(mean_height)
    # print(mean_width)
    
    # Resizing of the images to give
    # them same width and height 
    for file in os.listdir('.'):
        if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith("png"):
            # opening image using PIL Image
            im = Image.open(os.path.join(path, file)) 
    
            # im.size includes the height and width of image
            width, height = im.size   
            print(width, height)
    
            # resizing 
            imResize = im.resize((mean_width, mean_height), Image.ANTIALIAS) 
            imResize.save( file, 'JPEG', quality = 95) # setting quality
            # printing each resized image name
            print(im.filename.split('\\')[-1], " is resized") 
  
  



if __name__ == "__main__":

    moods = ['Happy', 'Funny', 'Chill', 'Dramatic', 'Sad', 'Romantic', 'Serious' , 'Scary' , 'Peaceful']
    video_themes = ['Landscape', 'Nature', 'Sports', 'Food', 'Buildings', 'Art', 'Technology' , 'Roadtrip']


    path = "D:\Fall 2021\Deep Learning for Text Data - CSE 8803 DLT\PROJECT\SYNCPHONIC\places365\images/trek"

    scene_info = []
    overall_theme = {}

    for image_path in os.listdir(path):
        input_path = os.path.join(path, image_path)

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

    video_preprocess(path)
    generate_video(path)

    mood = "upbeat"
    # path = "D:\Fall 2021\Deep Learning for Text Data - CSE 8803 DLT\PROJECT\SYNCPHONIC\places365\images\city"

    # final_theme = "buildings"
    # mood = "upbeat"
    add_audio(path, final_theme, mood)






    
    






    

    

