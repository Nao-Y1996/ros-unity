import sys
import cv2

images_path = 'webcamera_pose_data'
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
video = cv2.VideoWriter(images_path+'/pose.mp4',fourcc, 20.0, (640, 480))

if not video.isOpened():
    print("can't be opened")
    sys.exit()
i = 0
while True:
    try:
        img = cv2.imread(images_path+'/images/%d.png' % i)

        if img is None:
            print("can't read")
            break
        video.write(img)
        print(i)
        i += 1
    except:
        break

video.release()
print('written')
