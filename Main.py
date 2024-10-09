import numpy as np

t = np.linspace(-10, 20, 1000)
array1 = [1,2,3,4]
array2 = [8,7,6,5]
size = []

#Pour faire une convolution cov
result0 = np.convolve(array1, array2, mode='full')

#Pour faire une Cross-correaltion xcorr
result1 = np.correlate(array1, array2, mode='full')
print("resultat 1 = ",result1)

#Pour faire une heaviside
result2 = np.heaviside(t, 0.5)

#Fast Fourier Transform fft
result = np.fft.fft(array1)

#Uniform random number generation rand
result = np.random.rand(size) # Génère des nombres uniformes entre 0 et 1

#randn
result = np.random.randn(size) # Génère des nombres aléatoires selon une distribution normale standard