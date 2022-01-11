import pyaudio
import struct
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from tkinter import TclError

class AudioStream():

    def __init__(self):
        self.CHUNK = 1024 * 2
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.mic = pyaudio.PyAudio()
        self.stream = self.mic.open(
                        format=self.FORMAT,
                        channels=self.CHANNELS,
                        rate=self.RATE,
                        input=True,
                        output=True,
                        frames_per_buffer=self.CHUNK,
                        input_device_index=2)
        self.init_plots()
        self.start_plot()

    def init_plots(self):
        self.fig, (ax, ax2) = plt.subplots(2, figsize=(15, 8))

        # variable for plotting
        x = np.arange(0, 2*self.CHUNK, 2)
        x_fft = np.linspace(0, self.RATE, self.CHUNK)

        # create a line object with random data
        self.line, = ax.plot(x, np.random.rand(self.CHUNK), '-')
        self.line_fft, = ax2.semilogx(x_fft, np.random.rand(self.CHUNK), '-')

        # basic formatting for the axes
        ax.set_title('AUDIO WAVEFORM')
        ax.set_xlabel('samples')
        ax.set_ylabel('volume')
        ax.set_ylim(-3300, 3300)
        ax.set_xlim(0, 2*self.CHUNK)
        plt.setp(ax, xticks=[0, self.CHUNK, 2*self.CHUNK], yticks=[0])

        # format spectrum axes
        ax2.set_xlim(20, self.RATE/2)

        # show the plot
        plt.show(block=False)
        print('Stream started')


    def start_plot(self):
        # for measuring frame rate
        frame_count = 0
        start_time = time.time()

        while True:
            data = self.stream.read(self.CHUNK, exception_on_overflow=False)
        #         data_int = np.array(struct.unpack(str(2*CHUNK) + 'B', data), dtype='b')[::2]
            data_int = struct.unpack(str(self.CHUNK) + 'h', data)
            data_np = np.array(data_int, dtype=np.int16)
            self.line.set_ydata(data_np)
            
            y_fft = fft(data_int)
            self.line_fft.set_ydata(np.abs(y_fft[0:self.CHUNK]) * 2 / (127 * self.CHUNK))
                
            try:
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
                frame_count += 1
            except Exception:
                print('All is good')
                frame_rate = frame_count / (time.time() - start_time)
                print(f'Average frame rate: {round(frame_rate)} FPS')
                break
    

if __name__ == '__main__':
    AudioStream()