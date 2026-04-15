import numpy as np
import pandas as pd
from scipy.stats import norm
import sklearn
import matplotlib.pyplot as plt

np.random.seed(42)

#диапазон исследуемых длин волн
x=np.linspace(600, 800, 1024)


def gaus(x, A, mean, std):
	return A*np.exp((-(x-mean)**2)/(std**2))

# наши "пики" в спектре или сигнал который мы получили
peak_a = gaus(x, 150, 680, 15)
peak_b = gaus(x, 60, 650, 5)

clean_signal = peak_a + peak_b
# создание шумов и фонов
bckground = 5 + 0.02 * (x - 600) + 0.0005 * (x - 600)**2
bckground[438:500] += 13
noise = np.random.normal(0, 3, len(x))
poisson_noise = np.random.normal(0, np.sqrt(clean_signal + bckground + 1))

#результируй спект который мы получили
raw_spectrum = clean_signal + bckground + noise + poisson_noise
raw_spectrum = np.maximum(raw_spectrum, 0)

df = pd.DataFrame({"Wavelength_nm": x, "Intensity": raw_spectrum})
df.to_csv("spectrum_for_analysis.csv", index=False)

plt.figure(figsize=(12,5))
plt.plot(x, raw_spectrum, 'b-', alpha=0.6, label="Экспериментальные данные (грязные)")
plt.plot(x, clean_signal, 'r--', linewidth=2, label="Истинный сигнал (неизвестен)")
plt.title("Входящие данные: Нужно выделить вклад дефекта")
plt.xlabel("Длина волны (nm)")
plt.ylabel("Интенсивность")
plt.legend()
plt.grid(True)
plt.show()
