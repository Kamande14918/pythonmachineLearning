import scipy.signal as signal
import matplotlib.pyplot as plt
import numpy as np

# Specifications
wp = 1  # Passband edge frequency (rad/s)
ws = 2  # Stopband edge frequency (rad/s)
gstop = 40  # Stopband attenuation (dB)

# Design the Chebyshev Type II filter
n, wn = signal.cheb2ord(wp, ws, gpass=0.1, gstop=gstop, analog=True)
b, a = signal.cheby2(n, gstop, wn, btype='low', analog=True)

# Frequency response
w, h = signal.freqs(b, a, worN=np.logspace(-1, 2, 500))

# Plot the magnitude response
plt.figure()
plt.semilogx(w, 20 * np.log10(abs(h)))
plt.title('Chebyshev Type II Low-Pass Filter')
plt.xlabel('Frequency (rad/s)')
plt.ylabel('Magnitude (dB)')
plt.grid()
plt.axvline(wp, color='green', linestyle='--', label='Passband Edge')
plt.axvline(ws, color='red', linestyle='--', label='Stopband Edge')
plt.legend()
plt.show()