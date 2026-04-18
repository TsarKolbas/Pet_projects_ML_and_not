from scipy.optimize import curve_fit
from sklearn.metrics import r2_score


# 1. Определяем модель: сумма двух гауссиан
def double_gaussian(x, A1, mu1, sigma1, A2, mu2, sigma2):
    return (A1 * np.exp(-(x - mu1)**2 / (2 * sigma1**2)) +
            A2 * np.exp(-(x - mu2)**2 / (2 * sigma2**2)))

# 2. Начальные приближения (их нужно подобрать визуально)
# Смотрим на график: первый пик ~680 нм, амплитуда ~100, ширина ~15
# Второй пик ~650 нм, амплитуда ~40, ширина ~8
p0 = [120, 680, 14,     # A1, mu1, sigma1
      45, 650, 6]       # A2, mu2, sigma2

# 3. Границы параметров (чтобы алгоритм не "улетел")
# Нижние границы: [A1_min, mu1_min, sigma1_min, A2_min, mu2_min, sigma2_min]
# Верхние границы: [A1_max, mu1_max, sigma1_max, A2_max, mu2_max, sigma2_max]
bounds = ([0, 600, 1, 0, 600, 1],    # минимумы
          [300, 750, 40, 100, 700, 30])  # максимумы

# 4. Запускаем фит
popt, pcov = curve_fit(double_gaussian, x, spectrum_no_bcg, p0=p0, bounds=bounds)

# 5. Достаём результаты
A1, mu1, sigma1, A2, mu2, sigma2 = popt

# 6. Ошибки (из ковариационной матрицы)
perr = np.sqrt(np.diag(pcov))
err_A1, err_mu1, err_sigma1, err_A2, err_mu2, err_sigma2 = perr

print(f"Пик А: амплитуда = {A1:.2f} ± {err_A1:.2f}, центр = {mu1:.2f} ± {err_mu1:.2f}")
print(f"Пик Б: амплитуда = {A2:.2f} ± {err_A2:.2f}, центр = {mu2:.2f} ± {err_mu2:.2f}")


y_fit = double_gaussian(x, *popt)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# График 1: Сырые данные и фон
axes[0].plot(x, raw_Inten, 'b-', alpha=0.5, label='Сырые данные')
axes[0].plot(x, backGround, 'r--', label='Оценённый фон (SNIP)')
axes[0].set_title('1. Вычитание фона')
axes[0].set_xlabel('Длина волны (нм)')
axes[0].set_ylabel('Интенсивность')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# График 2: Очищенный спектр и аппроксимация
axes[1].plot(x, cleaned_df, 'b-', alpha=0.7, label='Очищенный спектр')
axes[1].plot(x, y_fit, 'r--', linewidth=2, label='Аппроксимация')
axes[1].fill_between(x, cleaned_df, y_fit, alpha=0.2, color='gray')
r2 = r2_score(cleaned_df, y_fit)
axes[1].set_title(f'2. Аппроксимация (R² = {r2:.4f})')
axes[1].set_xlabel('Длина волны (нм)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# График 3: Разделение пиков
axes[2].plot(x, cleaned_df, 'k-', alpha=0.5, label='Данные')
axes[2].plot(x, gaus(x, A1, mu1, sigma1), 'g-', label=f'Пик А: {A1:.1f} @ {mu1:.1f} нм')
axes[2].plot(x, gaus(x, A2, mu2, sigma2), 'r-', label=f'Пик Б: {A2:.1f} @ {mu2:.1f} нм')
axes[2].set_title('3. Разделение пиков')
axes[2].set_xlabel('Длина волны (нм)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

