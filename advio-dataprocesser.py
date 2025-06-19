import cv2
import os

seq_frame = {'01': ['000', '0015589'],
'02': ['000', '0011928'],
'03': ['000', '008846'],
'04': ['000', '007023'],
'05': ['000', '009669'],
'06': ['000', '0016536'],
'07': ['000', '005717'],
'08': ['000', '006412'],
'09': ['000', '005781'],
'10': ['000', '007371'],
'11': ['000', '0012396'],
'12': ['000', '006881'],
'13': ['000', '009099'],
'14': ['000', '006746'],
'15': ['000', '003105'],
'16': ['000', '008778'],
'17': ['000', '009448'],
'18': ['000', '0011826'],
'19': ['000', '008523'],
'20': ['000', '0018151'],
'21': ['000', '0019039'],
'22': ['000', '0023141'],
'23': ['000', '0012336'],}
frame_csv_path = ['ADVIO/advio-01/' , 'ADVIO/advio-02/' ,'ADVIO/advio-03/' ]
dir_id = ['01', '02','03']


from moviepy.editor import VideoFileClip

def getFrame(sec, folder_name):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)  # Go to the sec. position
    hasFrames,image = vidcap.read() # Retrieves the frame at the specified second
    if hasFrames:
      img_name = '{:010d}.png'.format(count)
      img_path = '{}/{}'.format(folder_name, img_name)
      cv2.imwrite(img_path, image) # Saves the frame as an image
      print(img_path)
    return hasFrames

for i in range(3):
    path = frame_csv_path[i]
    folder_name = path+'{}'.format(dir_id[i])
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(folder_name)
    # dir_path = 'ADVIO_frames/'
    clip = VideoFileClip(frame_csv_path[i]+'/iphone/frames.mov')
    print( clip.duration, "duration of clip" )
    vidcap = cv2.VideoCapture(frame_csv_path[i]+'/iphone/frames.mov')
    success,image = vidcap.read()
    count = 0 
    fps = 1/vidcap.get(cv2.CAP_PROP_FPS)
    print(fps, "fps")
    total_frames = 60*clip.duration
    print("total frames", total_frames)

    sec = 0
    count= 1
    success = getFrame(sec, folder_name)
    while success:
        count = count + 1
        sec = sec + fps
        sec = round(sec, 2)
        success = getFrame(sec,folder_name)

