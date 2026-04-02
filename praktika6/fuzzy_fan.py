import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# =========================
# ФУНКЦИИ ПРИНАДЛЕЖНОСТИ
# =========================

def triangular(x, a, b, c):
    """
    Треугольная функция принадлежности.
    Поддерживает как скаляр, так и numpy-массив.
    """
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)

    # Левая сторона
    if a != b:
        left = (x >= a) & (x <= b)
        y[left] = (x[left] - a) / (b - a)
    else:
        y[x <= b] = 1.0

    # Правая сторона
    if b != c:
        right = (x >= b) & (x <= c)
        y[right] = np.maximum(y[right], (c - x[right]) / (c - b))
    else:
        y[x >= b] = 1.0

    y[x == b] = 1.0
    y = np.clip(y, 0, 1)
    return y


# =========================
# НЕЧЁТКИЙ КОНТРОЛЛЕР
# =========================

class FuzzyFanController:
    def __init__(self):
        # Диапазоны
        self.temp_range = np.linspace(0, 50, 501)
        self.hum_range = np.linspace(0, 100, 1001)
        self.fan_range = np.linspace(0, 100, 1001)

        # Термы температуры
        self.temp_low = lambda x: triangular(x, 0, 0, 25)
        self.temp_medium = lambda x: triangular(x, 15, 25, 35)
        self.temp_high = lambda x: triangular(x, 30, 50, 50)

        # Термы влажности
        self.hum_low = lambda x: triangular(x, 0, 0, 60)
        self.hum_high = lambda x: triangular(x, 40, 100, 100)

        # Термы скорости вентилятора
        self.fan_low = lambda x: triangular(x, 0, 0, 50)
        self.fan_medium = lambda x: triangular(x, 25, 50, 75)
        self.fan_high = lambda x: triangular(x, 50, 80, 100)
        self.fan_very_high = lambda x: triangular(x, 75, 100, 100)

    def infer(self, temperature, humidity):
        """
        Выполняет нечёткий вывод и возвращает:
        - чёткое значение скорости
        - агрегированную функцию принадлежности по выходу
        - степени принадлежности входов
        """

        # Фаззификация входов
        mu_temp_low = float(self.temp_low(temperature))
        mu_temp_medium = float(self.temp_medium(temperature))
        mu_temp_high = float(self.temp_high(temperature))

        mu_hum_low = float(self.hum_low(humidity))
        mu_hum_high = float(self.hum_high(humidity))

        # Правила
        # 1. ЕСЛИ Температура = Низкая И Влажность = Низкая ТО Скорость = Низкая
        rule1 = min(mu_temp_low, mu_hum_low)

        # 2. ЕСЛИ Температура = Средняя И Влажность = Низкая ТО Скорость = Средняя
        rule2 = min(mu_temp_medium, mu_hum_low)

        # 3. ЕСЛИ Температура = Высокая ИЛИ Влажность = Высокая ТО Скорость = Высокая
        rule3 = max(mu_temp_high, mu_hum_high)

        # 4. ЕСЛИ Температура = Высокая И Влажность = Высокая ТО Скорость = Очень высокая
        rule4 = min(mu_temp_high, mu_hum_high)

        # Активизация заключений
        out_low = np.minimum(rule1, self.fan_low(self.fan_range))
        out_medium = np.minimum(rule2, self.fan_medium(self.fan_range))
        out_high = np.minimum(rule3, self.fan_high(self.fan_range))
        out_very_high = np.minimum(rule4, self.fan_very_high(self.fan_range))

        # Агрегация
        aggregated = np.maximum.reduce([out_low, out_medium, out_high, out_very_high])

        # Дефаззификация (центр тяжести)
        if np.sum(aggregated) == 0:
            crisp_output = 0.0
        else:
            crisp_output = np.sum(self.fan_range * aggregated) / np.sum(aggregated)

        fuzz_info = {
            "temp_low": mu_temp_low,
            "temp_medium": mu_temp_medium,
            "temp_high": mu_temp_high,
            "hum_low": mu_hum_low,
            "hum_high": mu_hum_high,
            "rule1": rule1,
            "rule2": rule2,
            "rule3": rule3,
            "rule4": rule4
        }

        return crisp_output, aggregated, fuzz_info

    def plot_membership_functions(self):
        # Температура
        plt.figure(figsize=(10, 5))
        plt.plot(self.temp_range, self.temp_low(self.temp_range), label='Low')
        plt.plot(self.temp_range, self.temp_medium(self.temp_range), label='Medium')
        plt.plot(self.temp_range, self.temp_high(self.temp_range), label='High')
        plt.title('Функции принадлежности: Temperature')
        plt.xlabel('Температура, °C')
        plt.ylabel('Степень принадлежности')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Влажность
        plt.figure(figsize=(10, 5))
        plt.plot(self.hum_range, self.hum_low(self.hum_range), label='Low')
        plt.plot(self.hum_range, self.hum_high(self.hum_range), label='High')
        plt.title('Функции принадлежности: Humidity')
        plt.xlabel('Влажность, %')
        plt.ylabel('Степень принадлежности')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Скорость вентилятора
        plt.figure(figsize=(10, 5))
        plt.plot(self.fan_range, self.fan_low(self.fan_range), label='Low')
        plt.plot(self.fan_range, self.fan_medium(self.fan_range), label='Medium')
        plt.plot(self.fan_range, self.fan_high(self.fan_range), label='High')
        plt.plot(self.fan_range, self.fan_very_high(self.fan_range), label='VeryHigh')
        plt.title('Функции принадлежности: Fan Speed')
        plt.xlabel('Скорость вентилятора, %')
        plt.ylabel('Степень принадлежности')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_output_for_input(self, temperature, humidity):
        crisp, aggregated, info = self.infer(temperature, humidity)

        plt.figure(figsize=(10, 5))
        plt.plot(self.fan_range, self.fan_low(self.fan_range), '--', label='Low')
        plt.plot(self.fan_range, self.fan_medium(self.fan_range), '--', label='Medium')
        plt.plot(self.fan_range, self.fan_high(self.fan_range), '--', label='High')
        plt.plot(self.fan_range, self.fan_very_high(self.fan_range), '--', label='VeryHigh')
        plt.fill_between(self.fan_range, aggregated, alpha=0.4, label='Aggregated Output')
        plt.axvline(crisp, linestyle=':', label=f'COG = {crisp:.2f}%')
        plt.title(f'Выход системы при T={temperature}°C, H={humidity}%')
        plt.xlabel('Скорость вентилятора, %')
        plt.ylabel('Степень принадлежности')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        print(f"Температура: {temperature} °C")
        print(f"Влажность:   {humidity} %")
        print(f"Скорость вентилятора: {crisp:.2f} %")
        print("Степени принадлежности и активации правил:")
        for k, v in info.items():
            print(f"  {k}: {v:.3f}")

    def simulation_table(self):
        """
        Тестовые данные по аналогии с заданием.
        """
        tests = [
            (20, 40, 30),
            (30, 50, 50),
            (40, 80, 80),
            (45, 90, 95)
        ]

        print("\nРЕЗУЛЬТАТЫ МОДЕЛИРОВАНИЯ")
        print("-" * 78)
        print(f"{'Темп.':>8} {'Влажн.':>8} {'Ожид., %':>12} {'Факт., %':>12} {'Ошибка, %':>12}")
        print("-" * 78)

        for t, h, expected in tests:
            actual, _, _ = self.infer(t, h)
            error = abs(expected - actual)
            print(f"{t:>8.1f} {h:>8.1f} {expected:>12.1f} {actual:>12.2f} {error:>12.2f}")

        print("-" * 78)

    def plot_3d_surface(self):
        temp_vals = np.linspace(0, 50, 41)
        hum_vals = np.linspace(0, 100, 41)

        T, H = np.meshgrid(temp_vals, hum_vals)
        Z = np.zeros_like(T)

        for i in range(T.shape[0]):
            for j in range(T.shape[1]):
                Z[i, j], _, _ = self.infer(T[i, j], H[i, j])

        fig = plt.figure(figsize=(11, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(T, H, Z, edgecolor='none', alpha=0.9)

        ax.set_title('3D-поверхность нечёткого вывода')
        ax.set_xlabel('Температура, °C')
        ax.set_ylabel('Влажность, %')
        ax.set_zlabel('Скорость вентилятора, %')

        plt.tight_layout()
        plt.show()


# =========================
# ОСНОВНАЯ ЧАСТЬ
# =========================

if __name__ == "__main__":
    controller = FuzzyFanController()

    # 1. Графики функций принадлежности
    controller.plot_membership_functions()

    # 2. Пример моделирования для одного набора входных данных
    controller.plot_output_for_input(temperature=40, humidity=80)

    # 3. Таблица результатов
    controller.simulation_table()

    # 4. 3D-поверхность вывода
    controller.plot_3d_surface()