import math as m
import numpy as np
import matplotlib.pyplot as plt
import csv
import random
import timeit
from numba import njit, types, typed, prange
import copy
from time import perf_counter


fig, ax = plt.subplots()

num_simul = 90
time_step = 1e-5  # параметры модели
display_step = int(1/time_step / 1000)
num_cycles = 90
num_spectra_points = 100
# density_matrix: eg -> 1,2

delta = 0
Rabi_freq = 2 * np.pi  # 1 MHz
delta_freq = 500.0
central_frequency = 72#360.0095

freq = []
intens = []
#spec_centre.csv 
with open(r"C:\Users\А\Desktop\Учеба\Научка\Шум лазера\servo_blue_seed\spectra_19.csv", newline='') as csvfile:  # ampl_noise\red_wide2_5av.csv
    spamreader = csv.reader(csvfile, delimiter=',')
    data_flag = 0
    for row in spamreader:
        if (data_flag):
            if (float(row[1]) > -80):
                freq.append(float(row[0]))
                intens.append(float(row[1]))
            #elif
        elif (row[0] == 'DATA'):
            print('YES')
            data_flag = 1

freqs = np.array(freq) / 1e6 - central_frequency + delta_freq
print('После отсекания осталось ', len(freqs), " частот")
signal = np.power(10, np.array(intens)/10)
signal = signal / np.sum(signal) #np.sum(signal[145:165])
amplitudes = (signal)**0.5
plt.scatter(freqs, signal)#[145:165])
plt.show()

mod_freqs = []
Rabi_amplitudes = []
for i in range(int(num_spectra_points/2.1)):
    mod_freqs.append(freqs[i * (len(freqs) // num_spectra_points)])
    Rabi_amplitudes.append(amplitudes[i * (len(freqs) // num_spectra_points)])
    mod_freqs.append(freqs[-i * (len(freqs) // num_spectra_points)])
    Rabi_amplitudes.append(amplitudes[-i * (len(freqs) // num_spectra_points)])

mod_freqs.append(delta_freq)
Rabi_amplitudes.append(1.0)

#main_peak = np.sum(signal[450:550])**0.5
'''servo_bump1 = np.sum(signal[100:180])**0.5
servo_bump2 = np.sum(signal[820:900])**0.5 #np.sum(signal[800:900])**0.5
#print(servo_bump1 / main_peak, servo_bump2 / main_peak, 'left = ', 1-main_peak**2-servo_bump1**2-servo_bump2**2 )
mod_freqs.append(freqs[140])
Rabi_amplitudes.append(servo_bump1)
mod_freqs.append(freqs[860])
Rabi_amplitudes.append(servo_bump2)'''


mod_freqs = np.array(mod_freqs)
Rabi_amplitudes = np.array(Rabi_amplitudes)

plt.scatter(mod_freqs, Rabi_amplitudes)
plt.show()

@njit(cache=True, nogil=True, fastmath=True, parallel=False)
def inv_fourier(amplitudes, freqs, time, phases):
    result = np.zeros(len(time), dtype = 'complex_')
    k = len(freqs)
    for i in range(len(time)):
        args = time[i] * freqs + phases
        result[i] = np.sum(amplitudes * (np.cos(args) + 1j * np.sin(args)))
    return result


@njit(cache=True, nogil=True, fastmath=True, parallel=False)
def model_evolution(num_points_in_model, coeffs, initial_state):
    #phi = 2 * np.pi * (1000 * rd_phase - int(1000 * rd_phase))
        #d_rho = np.array([0+0j,0+0j,0+0j,0+0j])
    # display_num_p22 = np.array([0.0]*num_points_in_model // display_step)#np.zeros(num_points_in_model // display_step, dtype=float)
    # print(num_points_in_model, display_step)
    #t = perf_counter()
    #_coeffs = np.sum(coeffs * (np.cos(rd_phases) + 1j * np.sin(rd_phases)), axis=-1)
    #_coeffs = coeffs        
    #print('Adding phases', perf_counter() - t)
    length_result = int(num_points_in_model/display_step) + 1
    #length_result2 = 2 * length_result
    # [0.0]*num_points_in_model
    display_num = np.zeros(length_result, dtype=np.float64)
    '''display_num_1 = np.zeros(num_points_in_model//display_step + 1, dtype=np.float64)
    display_num_2 = np.zeros(num_points_in_model//display_step + 1, dtype=np.float64)
    display_num_3 = np.zeros(num_points_in_model//display_step + 1, dtype=np.float64)'''
    #Rabi_freq_div2 = Rabi_freq / 2
    '''servo_rel_ampl_1 = 4 * servo_bump1 / main_peak
    servo_rel_ampl_2 = 2 * servo_bump2 / main_peak'''
    display_count = 0

    state0_init = initial_state[0]
    state1_init = initial_state[1]
    state_init = state0_init + state1_init
    state2 = initial_state[2]
    res = 0+0j

    for i in prange(0, num_points_in_model):
        #t = perf_counter()
        coeff = coeffs[i]
        res += -1j * coeff * state2
        state2 += -1j * np.conj(coeff) * (2 * res + state_init)#state0_init + state1_init)

        #print(state[0], state[1], state[2], int_Rabi_spectra_1)

        if (i % display_step == 0):
            display_num[display_count] = np.abs(res+state1_init)**2
            #display_num[length_result + display_count] = abs(state[1])**2
            #display_num[length_result2 + display_count] = abs(state[2])**2

            '''display_num_1[display_count] = abs(state[0])**2
            display_num_2[display_count] = abs(state[1])**2
            display_num_3[display_count] = abs(state[2])**2'''
            display_count += 1

        # if (i % num_cycle == 0):
        #    print(i // num_cycle)

        #print(perf_counter() - t)

    # print(rho_0)
    #result = np.array([display_num_1,display_num_2,display_num_3])
    return display_num


state_0 = np.array([1.0+0.0j, 0.0+0.0j, 0.0+0.0j])

#rd_phases = 2 * np.pi * (np.random.random_sample(num_simul))

rd_phases = 2*np.pi * np.random.random_sample((len(mod_freqs),))
num_points_in_model = int(
    num_cycles*2*np.pi/((Rabi_freq**2+delta**2)**0.5 * time_step))
noise_scale_factor = 1.5
#num_cycle = int(2*np.pi/((Rabi_freq**2+delta**2)**0.5 * time_step))

t = perf_counter()

time = np.arange(0, num_points_in_model) * time_step
#time, _ = np.meshgrid(np.ones_like(mod_freqs), time)
'''arg = time * mod_freqs
Rabi_t = time_step * Rabi_freq / 2 * Rabi_amplitudes * (np.cos(arg) + 1j * np.sin(arg))'''
Rabi_t = noise_scale_factor * Rabi_freq/2 * time_step * inv_fourier(Rabi_amplitudes, mod_freqs, time, rd_phases)
print("Coeffs", perf_counter() - t)
#plt.plot(time, np.real(Rabi_t)/time_step)
#plt.plot(time, np.imag(Rabi_t)/time_step)
#plt.show()
#print(timeit.timeit('model_evolution(copy.deepcopy(state_0), time_step, Rabi_freq, amplitudes, freqs, rd_phases[0])', globals=globals(), number = 10))

t = perf_counter()
display = model_evolution(num_points_in_model, Rabi_t, copy.deepcopy(state_0))
print("Step", perf_counter() - t)

timeline = np.linspace(0, num_cycles, len(display))
plt.plot(timeline, (1-np.cos(Rabi_freq**2/2/delta_freq * timeline))/2)
plt.plot(timeline, display, color='red', linewidth=0.5)

for i in range(num_simul-1):
    print(i+1)
    noise_scale_factor = 0.97 * noise_scale_factor
    rd_phases = 2*np.pi * np.random.random_sample((len(mod_freqs),))
    t = perf_counter()
    Rabi_t = noise_scale_factor * Rabi_freq/2 * time_step * inv_fourier(Rabi_amplitudes, mod_freqs, time, rd_phases)
    result = model_evolution(num_points_in_model, Rabi_t, copy.deepcopy(state_0))
    print("Step", perf_counter() - t)
    display += result
    plt.plot(timeline, result, color='red', linewidth=0.5)

plt.plot(timeline, display/num_simul, color='blue',
         label=f'step = {time_step}, num_simul = {num_simul}, noise_pts = {num_spectra_points}')
plt.legend()
plt.show()
'''
spec_res = np.fft.fft(display)#/np.mean(np.array(intens))
freq_res = np.fft.fftfreq(len(display), d=time_step)
plt.plot(freq_res, np.abs(spec_res)/len(spec_res))
plt.show()'''
'''
for i in range(1,len(amplitudes)):
    print(i)
    rd_phases = 2*np.pi * np.random.random_sample((1,)) 
    result = model_evolution(copy.deepcopy(state_0), time_step, Rabi_freq, amplitudes[i], freqs[i], rd_phases)
    plt.plot(timeline, result, color = 'red', linewidth = 0.5)
plt.show()
'''

'''result = model_evolution(copy.deepcopy(state_0), time_step, Rabi_freq, amplitudes, freqs, rd_phases)
#result_wo_numba = model_evolution_wo_numba(copy.deepcopy(state_0), time_step, Rabi_freq, amplitudes, freqs, rd_phases)
#print(timeit.timeit('model_evolution(copy.deepcopy(state_0), time_step, Rabi_freq, amplitudes, freqs, rd_phases)', globals=globals(), number = 10))

timeline = np.linspace(0, num_cycles, len(result)//3)
plt.plot(timeline, result[:len(result)//3], label = 'ground')
plt.plot(timeline, result[len(result)//3 : 2 * len(result)//3], label = 'excited')
#plt.plot(timeline, result[2 * len(result)//3:], label = 'rydberg')
#plt.plot(timeline, result_wo_numba[:len(result_wo_numba)//3], label = 'ground_wo_numba')
#plt.plot(timeline, result_wo_numba[len(result_wo_numba)//3 : 2 * len(result_wo_numba)//3], label = 'excited_wo_numba')
#plt.plot(timeline, result_wo_numba[2 * len(result_wo_numba)//3:], label = 'rydberg_wo_numba')
plt.legend()
plt.show()
'''
