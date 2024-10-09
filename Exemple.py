import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve, correlate

# Définir les paramètres de base
a = 0.5  # Paramètre de décroissance du signal exponentiel
t = np.linspace(-10, 20, 1000)  # Intervalle de temps

# a) Définir les deux signaux h(t) et x(t)
# Signal h(t) : signal porte (carré) de largeur 10 dans l'intervalle [-5, 5]
h = np.where((t >= -5) & (t <= 5), 1, 0)

# Signal x(t) = 2e^(-at) * u(t), où u(t) est l'échelon unitaire
u = np.heaviside(t, 1)  # Fonction échelon
x = 2 * np.exp(-a * t) * u  # Signal exponentiel

# Calculer la convolution de x(t) et h(t)
y = convolve(x, h, mode='same') * (t[1] - t[0])  # 'same' pour conserver la même longueur que t

# Affichage des signaux
plt.figure(figsize=(12, 8))

# Signal h(t)
plt.subplot(3, 1, 1)
plt.plot(t, h, label='h(t) : Signal carré [−5,5]', color='blue')
plt.title('Signal h(t) (Carré)')
plt.xlabel('Temps')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()

# Signal x(t)
plt.subplot(3, 1, 2)
plt.plot(t, x, label='x(t) = 2e^(-at).u(t)', color='green')
plt.title('Signal x(t) (Exponentiel)')
plt.xlabel('Temps')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()

# Convolution y(t) = x(t) * h(t)
plt.subplot(3, 1, 3)
plt.plot(t, y, label='y(t) = x(t) * h(t)', color='red')
plt.title('Produit de Convolution y(t)')
plt.xlabel('Temps')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# b) Inter-corrélation entre x(t) et h(t)
inter_corr = correlate(x, h, mode='same')

# Affichage de l'inter-corrélation
plt.figure(figsize=(6, 4))
plt.plot(t, inter_corr, label='Inter-corrélation de x(t) et h(t)', color='purple')
plt.title('Inter-corrélation entre x(t) et h(t)')
plt.xlabel('Temps')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()
plt.show()

# c) Calcul de l'autocorrélation d'un signal cosinus
# Définir un signal cosinus de fréquence quelconque
f = 1  # Fréquence du cosinus
cos_signal = np.cos(2 * np.pi * f * t)

# Calcul de l'autocorrélation
auto_corr = correlate(cos_signal, cos_signal, mode='same')

# Affichage du signal cosinus et de son autocorrélation
plt.figure(figsize=(12, 6))

# Signal cosinus
plt.subplot(2, 1, 1)
plt.plot(t, cos_signal, label='Signal cosinus', color='orange')
plt.title('Signal Cosinus')
plt.xlabel('Temps')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()

# Autocorrélation
plt.subplot(2, 1, 2)
plt.plot(t, auto_corr, label='Autocorrélation du signal cosinus', color='magenta')
plt.title('Autocorrélation du Signal Cosinus')
plt.xlabel('Temps')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
