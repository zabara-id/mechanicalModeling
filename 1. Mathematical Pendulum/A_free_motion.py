import numpy as np
from matplotlib import pyplot as plt


# Функция правых частей динамической системы "математический маятник"
def RS_pendulum(theta: np.ndarray, omega: np.ndarray, t, g, l) -> tuple:
    return omega, -g/l*np.sin(theta)


# шаг РК4
def RK4_step(func, dt, theta, omega, t, g, l):
    k1_theta, k1_omega = func(theta, omega, t, g, l)
    k2_theta, k2_omega = func(theta + dt/2*k1_theta, omega + dt/2*k1_omega, t + dt/2, g, l)
    k3_theta, k3_omega = func(theta + dt/2*k2_theta, omega + dt/2*k2_omega, t + dt/2, g, l)
    k4_theta, k4_omega = func(theta + dt*k3_theta, omega + dt*k3_omega, t + dt, g, l)

    theta += dt/6*(k1_theta + 2*k2_theta + 2*k3_theta + k4_theta)
    omega += dt/6*(k1_omega + 2*k2_omega + 2*k3_omega + k4_omega)
    E = 0.5 * m * l**2 * omega0**2 + m * g * l * (1 - np.cos(theta0))

    return theta, omega, E


# Начальные данные
theta0 = 0.1  # начальный угол
omega0 = 0.0  # начальная угловая скорость
g = 9.8  # ускорение свободного падения
l = 1.0  # длина маятника
m = 1.0  # масса маятника
E0 = 0.5 * m * l**2 * omega0**2 + m * g * l * (1 - np.cos(theta0))

# Параметры времени
t0 = 0  # начальное время
T = 10  # конечное время
dt = 0.01  # шаг по времени
t = np.arange(t0, T, dt)

# Решение дифференциального уравнения
theta, omega, E = np.empty_like(t), np.empty_like(t), np.empty_like(t)
theta[0], omega[0], E[0] = theta0, omega0, E0

# РК4
for i in range(1, t.size):
    theta[i], omega[i], E[i] = RK4_step(RS_pendulum, dt, theta[i-1], omega[i-1], t[i-1], g, l)

plt.figure(figsize=(12, 8))

plt.subplot(221)
plt.plot(t, theta, color='b')
plt.title('Угол в зависимости от времени')
plt.xlabel('Время')
plt.ylabel('Угол')

plt.subplot(222)
plt.plot(t, omega, color='orange')
plt.title('Угловая скорость в зависимости от времени')
plt.xlabel('Время')
plt.ylabel('Угловая скорость')

plt.subplot(223)
plt.plot(theta, omega, color='r')
plt.title('Фазовая картина движения')
plt.xlabel('Угол')
plt.ylabel('Угловая скорость')

plt.subplot(224)
plt.plot(t, E, color='g')
plt.title('Полная энергия в зависимости от времени')
plt.xlabel('Время')
plt.ylabel('Энергия')

plt.tight_layout()
plt.show()
