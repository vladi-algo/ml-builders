import boto3
import os
import sys

polly_client = boto3.client('polly')
 
text_to_send = str(sys.argv[1])

response = polly_client.synthesize_speech(VoiceId='Matthew',
                                          Engine='neural',
                                          OutputFormat='mp3', 
                                          Text = text_to_send)

file = open('speech.mp3', 'wb')
file.write(response['AudioStream'].read())
file.close()

dir_path = os.path.dirname(os.path.realpath(__file__)).replace(" ", "\ ") + '/speech.mp3'

os.system(("afplay " + dir_path))