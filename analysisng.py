from scipy.ndimage import gaussian_filter1d

from scipy.signal import savgol_filter
#считаем данные
df = pd.read_csv('spectrum_for_analysis.csv')

#посмотрим, что в них
#df.info()
#df.shape

def snip_background_fast(y, iterations=70):
    """
    Быстрая векторизованная версия SNIP.
    """
    y = np.maximum(y, 1e-10)
    y_log = np.log(y)

    for i in range(iterations):
        # Создаем массив для нового значения
        y_new = y_log.copy()

        left = np.roll(y_log, 1)   # сдвиг вправо (i-1)
        right = np.roll(y_log, -1)  # сдвиг влево (i+1)
        avg_neighbors = (left + right) / 2

        y_new = np.minimum(y_log, avg_neighbors)
        y_log = y_new

    background = np.exp(y_log)
    return background

raw_Inten = df['Intensity'].values.copy()

backGround = snip_background_fast(raw_Inten)
spectrum_no_bcg = raw_Inten - backGround

plt.figure(figsize=(12,5))
plt.plot(df['Wavelength_nm'], raw_Inten, 'b-', alpha=0.6)
plt.title('без фона спектр')
plt.xlabel("Длина волны (nm)")
plt.ylabel("Интенсивность")
plt.legend()
plt.grid(True)
plt.show()



cleaned_df = savgol_filter(raw_Inten, 5, 2)
plt.figure(figsize=(12,5))
plt.plot(df['Wavelength_nm'], cleaned_df, 'b-', alpha=0.6)
plt.title('сглаженный спектр')
plt.xlabel("Длина волны (nm)")
plt.ylabel("Интенсивность")
plt.legend()
plt.grid(True)
plt.show()
