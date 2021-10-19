import pygame
import pygame._sdl2 as sdl2
import time



pygame.init()
is_capture = 0  # zero to request playback devices, non-zero to request recording devices
num = sdl2.get_num_audio_devices(is_capture)
names = [str(sdl2.get_audio_device_name(i, is_capture), encoding="utf-8") for i in range(num)]
print("\n".join(names))
pygame.quit()

'''
pygame.mixer.pre_init(devicename="Built-in Output")
pygame.mixer.init()
pygame.mixer.music.load("../speech.mp3")
pygame.mixer.music.play()
time.sleep(5)
'''

pygame.mixer.pre_init(devicename="BlackHole 16ch")
pygame.mixer.init()
pygame.mixer.music.load("../speech.mp3")
pygame.mixer.music.play()
time.sleep(10)


pygame.mixer.quit()

#from pygame._sdl2 import get_num_audio_devices, get_audio_device_name #Get playback device names
#from pygame import mixer #Playing sound

#l=get_audio_device_name(0)
#mixer.init() #Initialize the mixer, this will allow the next command to work
#devices=[get_audio_device_name(x, 0).decode() for x in range(get_num_audio_devices(0))] #Returns playback devices
#['Headphones (Oculus Virtual Audio Device)', 'MONITOR (2- NVIDIA High Definition Audio)', 'Speakers (High Definition Audio Device)', 'Speakers (NVIDIA RTX Voice)', 'CABLE Input (VB-Audio Virtual Cable)']

#print(devices)
#mixer.quit() #Quit the mixer as it's initialized on your main playback device
#mixer.init(devicename='CABLE Input (VB-Audio Virtual Cable)') #Initialize it with the correct device
#mixer.music.load("../speech.mp3") #Load the mp3
#mixer.music.play() #Play
