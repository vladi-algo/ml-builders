# 
# Usage:
# $ python3 polly_text_to_speech.py "my message free text comes here"
# 

import boto3
import os
import sys
import pyaudio
from contextlib import closing

SAMPLE_RATE = 16000
READ_CHUNK = 4096
CHANNELS = 1
BYTES_PER_SAMPLE = 2

polly_client = boto3.client('polly')
 
text_to_send = str(sys.argv[1])

response = polly_client.synthesize_speech(VoiceId='Matthew',
                                          Engine='neural',
                                          OutputFormat='pcm', 
                                          Text = text_to_send,
                                          SampleRate=str(SAMPLE_RATE))



p = pyaudio.PyAudio()
stream = p.open(format=p.get_format_from_width(BYTES_PER_SAMPLE),
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                output=True)

with closing(response["AudioStream"]) as polly_stream:
    while True:
        data = polly_stream.read(READ_CHUNK)
        if data is None:
            break

        stream.write(data)

stream.stop_stream()
stream.close()

p.terminate()