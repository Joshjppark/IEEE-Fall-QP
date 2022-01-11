import pyaudio
import struct
import math 
import time
import serial


# create constants for recording
FORMAT  = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
BLOCK_TIME = 0.02 # anaylze sound within frames of 0.02 seconds
FRAMES_PER_BLOCK = int(RATE*BLOCK_TIME) # equals 882; implies that a clap should be recognized within 882 data samples

# if we get this many noisy blocks in a row, increase the threshold
OVERSENSITIVE = 15.0/BLOCK_TIME                    
# if we get this many quiet blocks in a row, decrease the threshold
UNDERSENSITIVE = 120.0/BLOCK_TIME 

# threshold constants for claps
INIT_THRESHOLD = 0.25
MAX_CLAP_BLOCKS = 5

class ClapTester():
    
    def __init__(self, device_index):
        self.mic = pyaudio.PyAudio()
        self.stream = self.mic.open(
                            format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            input_device_index=device_index,
                            frames_per_buffer=FRAMES_PER_BLOCK
                            )
        self.clap_threshold = INIT_THRESHOLD
        self.noisycount = MAX_CLAP_BLOCKS + 1 # adding one so that the program does not register a clap on first listen() call
        self.quietcount = 0
        self.detected = False
        # self.increment_clap = False
    
    
    def get_rms(self, block):
        # RMS amplitude is the squart root of the mean of the squares of the ampltiude
        count = len(block)/2 # we divide by 2 to get the number of samples, since each sample is written with 2 bytes
        format = f"{int(count)}h"
        int_block = struct.unpack(format, block) # converts data into 16 bit integers
        
        # iterate over the block
        sum_squares = 0.0
        for sample in int_block:
            # sample is a signed 16 bit integer in +/- 32768
            n = sample * (1.0/32768.0) # normalizes the data
            sum_squares += n*n
            
        return math.sqrt(sum_squares/count)

        
    def stop(self):
        self.stream.close()

    def listen(self):
        block = self.stream.read(FRAMES_PER_BLOCK)
        amplitude = self.get_rms(block)
        if amplitude > self.clap_threshold:
            # the block is noisy:
            self.quietcount = 0
            self.noisycount += 1
            # if self.noisycount > OVERSENSITIVE:
            #     # turn down the sensitivity (threshold increases)
            #     self.clap_threshold *= 1.1
        else:
            # block is quiet
            if 1 <= self.noisycount <= MAX_CLAP_BLOCKS:
                # a clap is detected if it is noisy for a little bit
                self.clapDetected()
            self.noisycount = 0
            self.quietcount += 1
            # if self.quietcount > UNDERSENSITIVE:
            #     # turn up the sensitivity (threshold decreases)
            #     self.clap_threshold *= 0.9

        
    def clapDetected(self):
        self.detected = True
        # print(f'Clap')


def main():
    ser = serial.Serial('COM4', 9600, timeout=1)
    clap = ClapTester(device_index=1)
    leds_on = False
    num_claps = 0
    print('Start')
    start_time = time.time()
    try:
        while True:
            clap.listen()
            now = time.time()
            if(clap.detected):
                clap.detected = False
                num_claps += 1

                if(num_claps == 2):
                    print('DOUBLE Clap')
                    num_claps = 0

                    if leds_on: # leds are on; turn them off
                        ser.write(b'L')
                        leds_on = False
                    else: # leds are off; turn them on
                        ser.write(b'H')
                        leds_on = True

                else:
                    start_time = time.time()
                    print('SINGE Clap')
            if(round(now - start_time, 1)) > .5:
                num_claps = 0
            # print(round(now - start_time))

    except KeyboardInterrupt:
        clap.stop()
        ser.close()
        print('Finished')


if __name__ == '__main__':
    main()