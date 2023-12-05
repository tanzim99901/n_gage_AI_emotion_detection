import os


def MixAudioVideo(video_path, audio_path, output_path, idx, tot, filename):
    print("Mixing " + str(idx) + " / " + str(tot) + " : " + str(filename))
    video = VideoFileClip(video_path)
    fps = video.fps
    #print(fps)
    video = video.set_audio(AudioFileClip(audio_path))
    video.write_videofile(output_path)
    
directory = 'videos'

res = []

for file in os.listdir(directory):
    if file.endswith('.mp4'):
        if 'processed' not in file:
            tempName = file.replace('.mp4','_processed.mp4')
            if os.path.exists(directory + "//" + tempName):
                tempName2 = file.replace('.mp4','_processed_wSound.mp4')
                if not os.path.exists(directory + "//" + tempName2):
                    res.append(file)
                

print(len(res))

total_files = len(res)

for i in range(0, len(res)):
    vidPath = directory + "//" + res[i].replace('.mp4','_processed.mp4')
    audPath = directory + "//" + res[i].replace('.mp4','.wav')
    outPath = directory + "//" + res[i].replace('.mp4','_processed_wSound.mp4')
    
    MixAudioVideo(vidPath, audPath, outPath, i, total_files, res[i])
