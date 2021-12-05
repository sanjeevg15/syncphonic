import cv2
import os

vid_name = 'beach'
vid_path = 'videos/' + vid_name + '.mp4'

"""
if not os.path.isdir('images/' + vid_name):
    os.mkdir('images/' + vid_name)
"""

vid = cv2.VideoCapture(vid_path)
success, image = vid.read()

frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

skip = frame_count // 10

count = 0
while success:
    success, image = vid.read()
    #print('Read a new frame: ', success)

    if count % skip == 0:
        print("wrote image")
        cv2.imwrite("images/frame{}.jpg".format(count), image)  # save frame as JPEG file

    count += 1
