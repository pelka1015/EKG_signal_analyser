import csv
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import filedialog
from scipy import signal
#####################################################################################___Loading data___##########################################################################################################
def load_data():

    def csv_reader(file_path): 
        with open(file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile, delimiter='\n')  
            next(reader)
            for row in reader:
                if row:  
                    yield float(row[0])  

    def file_explorer():
        data_path = filedialog.askopenfilename(
            initialdir=r"DataSets\data.csv",  
            title="Choose a CSV file",
            filetypes=(("Input data", "*.csv"), ("All files", "*.*"))  
        )
        if data_path:
            return data_path    
        else:
            print("No chosen file.")

    if __name__ == "__main__": 
        root = tk.Tk()
        root.withdraw()  
        return csv_reader(file_explorer())

# Load data from CSV file
data = [x for x in load_data()]
print(len(data))
#################################################################################___Functions to creating plots___######################################################################################################
#Characteristic frequency response plot
plt.style.use(['dark_background'])
def plot_response(w, h, title, xlim):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(w/2, 20*np.log10(np.abs(h))) 
    plt.xlim(0, xlim)
    ax.set_ylim(-40, 5)
    ax.grid(True,color='grey')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Gain (dB)')
    ax.set_title(title)

#Draws two plots in time domain
def subplot2_s(signal1,signal2):
    t = np.linspace(0,len(signal1)*(1/256),len(signal1))

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.xlim(1000*(1/256),2000*(1/256))
    plt.plot(t,signal1)
    plt.title("Input signal")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.grid(True,color='grey')

    plt.subplot(2, 1, 2) 
    plt.xlim(1000*(1/256),2000*(1/256)) 
    plt.plot(t,signal2)
    plt.title("Output signal")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.grid(True,color='grey')

    plt.tight_layout()  
    plt.show()

#Draw s two plots in frequency domain
def subplot2_f(signal1, signal2, Fs=256):
    fft_result1 = np.fft.fft(signal1)
    fft_result2 = np.fft.fft(signal2)

    fft_magnitude1 = np.abs(fft_result1) / len(signal1)
    fft_magnitude2 = np.abs(fft_result2) / len(signal2)
    fft_freq = np.fft.fftfreq(len(signal1), 1 / Fs)

    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(fft_freq[:len(fft_freq)//2], fft_magnitude1[:len(fft_freq)//2], linestyle='-', marker='o', markersize=3)
    plt.xlim(0, Fs / 2)
    plt.title("Amplitude Spectrum before")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.grid(True, color='grey')

    plt.subplot(2, 1, 2)
    plt.plot(fft_freq[:len(fft_freq)//2], fft_magnitude2[:len(fft_freq)//2], linestyle='-', marker='o', markersize=3)
    plt.xlim(0, Fs / 2)
    plt.title("Amplitude Spectrum after")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.grid(True, color='grey')

    plt.tight_layout()
    plt.show()


#################################################################################___Filtry___#####################################################################################################################################
#upper cut-off frequency filter
fs = 300; # sampling frequency
cutoff = 1   # cut-off frequency
trans_width = 0.8  # width of the transition band
numtaps = 621   # number of filter taps (order + 1)
taps = signal.remez(numtaps, [0, cutoff - trans_width, cutoff, 0.5*fs], [0, 1], fs=fs)
w, h = signal.freqz(taps, [1], worN=2000, fs=fs)

signal_filtred = np.convolve(data, taps, mode="same")

plot_response(w, h, "High pass filter", 20)
subplot2_s(data,signal_filtred)
subplot2_f(data, signal_filtred)


#notch filter 50Hz
fs = 512 
band = [99, 101]  
trans_width = 2   
numtaps =  431
edges = [0, band[0] - trans_width, band[0], band[1], band[1] + trans_width, 0.5*fs]

taps = signal.remez(numtaps, edges, [1, 0, 1], fs=fs)
w, h = signal.freqz(taps, [1], worN=2000, fs=fs)

signal_filtred3 = np.convolve(signal_filtred, taps, mode="same")
signal_filtred3 = signal_filtred3[:len(data):]

plot_response(w, h, "Band stop filter", 120)
subplot2_s(signal_filtred, signal_filtred3)
subplot2_f(signal_filtred, signal_filtred3)
  
##################################################################################___Analysin EKG signal___#########################################################################################################################################
#Finding local maxima
maxima = -100
maximas = {} 
a = 0

for i in range(len(signal_filtred3)):
    if signal_filtred3[i] > maxima:
        maxima = signal_filtred3[i]
        a = 0
    elif maxima > signal_filtred3[i] and maxima > 0:
        a += 1
    if a == 40:
        maximas[(i - 40) * (1 / 256)] = maxima
        maxima = -100
        a = 0

#marking R, P, T peaks
"""
Method of operation:

We identify peaks that exceed the critical value of 0.5.
We assign to the list the peaks corresponding to r (only the r peaks exceed this critical value).
The maxima to the left of the r peak are assigned as p peaks.
The maxima to the right of the r peak are assigned as t peaks.
"""
r_peak = {}
p_peak = {}
t_peak = {}
# Creating a list of tuples from the maximas dictionary
maxima_list = list(maximas.items())

for index, (key, value) in enumerate(maximas.items()):
    if value > 0.5:
        if index - 1 >= 0:  
            key_p, value_p = maxima_list[index - 1]
            p_peak[key_p] = value_p  
        
        if index + 1 < len(maxima_list):  
            key_t, value_t = maxima_list[index + 1]
            t_peak[key_t] = value_t  
        
        r_peak[key] = value


#marking P wave

p_wave = {}  
delta_threshold = 1e-4
delta_threshold2 =  1e-2
a = 0

# Iteration over all P peaks
for key in p_peak.keys():
    index = int(key * 256)  # changing P_peak time to index
    
    # Backward iteration from P_peak index
    for j in range(index, 0, -1):
        # Calculating the derivative of the signal
        delta = abs(signal_filtred3[j] - signal_filtred3[j - 1])
        
        # checking if the derivative is less than the threshold value
        if delta < delta_threshold:
            break  
        else:
            p_wave[j/256] = signal_filtred3[j]  
        
    
# Iteration over all P peaks
for key in p_peak.keys():
    index = int(key * 256)  # changing P_peak time to index
    a = 0
    # Forward iteration from P_peak index
    for j in range(index, len(signal_filtred3) - 1):
        # Calculating the derivative of the signal
        delta = abs(signal_filtred3[j + 1] - signal_filtred3[j])
        
        # checking if the derivative is less than the threshold value
        if delta < delta_threshold2 and a > 30:
            break  
        else:
            p_wave[j / 256] = signal_filtred3[j]
            a+=1




#marking T wave
t_wave = {} 
delta_threshold = 1e-3
delta_threshold2 = 1e-2

for key in t_peak.keys():
    index = int(key * 256) 
    a = 0  
    
    for j in range(index, 0, -1):
        delta = abs(signal_filtred3[j] - signal_filtred3[j - 1])
        
        if delta < delta_threshold2 and a >= 40:
            break  
        else:
            t_wave[j / 256] = signal_filtred3[j]
            a += 1

for key in t_peak.keys():
    index = int(key * 256)  
    a = 0  
    
    for j in range(index, len(signal_filtred3) - 1):
        delta = abs(signal_filtred3[j + 1] - signal_filtred3[j])
        
        if delta < delta_threshold2 and a >= 40:
            break 
        else:
            t_wave[j / 256] = signal_filtred3[j]
            a += 1



#Marking Q peaks
delta_threshold = 1e-3*8
delta_threshold2 = 1e-3*8
q_peak = {} 
s_peak = {} 

for key in r_peak.keys():
    index = int(key * 256)  
    a = 0  
    
    for j in range(index, 0, -1):
        delta = abs(signal_filtred3[j] - signal_filtred3[j - 1])
        
        if delta < delta_threshold2 and a >= 2:
            q_peak[j / 256] = signal_filtred3[j] #To dictionary only the last value is added
            break  
        else:
            a += 1


#Marking S peaks
for key in r_peak.keys():
    index = int(key * 256)  
    a = 0  

    for j in range(index, len(signal_filtred3) - 1):
        delta = abs(signal_filtred3[j + 1] - signal_filtred3[j])
        
        if delta < delta_threshold2 and a >= 2:
            s_peak[j / 256] = signal_filtred3[j] #To dictionary only the last value is added
            break  
        else:
            
            a += 1


#Calculating heart rate
czestosc_bicia_serca = (len(r_peak)*60)/(len(data)*1/256)


#Plotting the marked peaks on the signal
t = np.linspace(0, len(data) * (1 / 256), len(data))  # time vector

plt.figure()
plt.xlim(1000*(1/256),2000*(1/256))
plt.plot(t, signal_filtred3,label="EKG signal after filtration")
plt.scatter(p_wave.keys(), p_wave.values(), color='lightblue', label="P")
plt.scatter(t_wave.keys(), t_wave.values(), color='lightgreen', label="T")
plt.scatter(q_peak.keys(), q_peak.values(), color='lime', label="Q")
plt.scatter(r_peak.keys(), r_peak.values(), color='red', label="R")  
plt.scatter(s_peak.keys(), s_peak.values(), color='yellow', label="S")
plt.figtext(0.05,0, f"Heart beat frequency: {int(round(czestosc_bicia_serca,0))} beats per minute", ha='left', fontsize=12, color='white')

plt.grid(True,color='grey')
plt.xlabel("Time [s]")
plt.ylabel("Signl value")
plt.title("Marked EKG signal peaks")
plt.legend()
plt.show()
