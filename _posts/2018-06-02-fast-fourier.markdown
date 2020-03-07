---
layout: post
date:   2017-11-01
status: published
image: yorubanames.png 
title: Generating Yoruba Names using Char-RNN
status: published
---


NOTE : This is the first of a two part blog post on the topic.

The Fast Fourier Transform is an algorithmic optimization of the Discrete Fourier Transform.

 I came across it a couple of weeks back and found it quite interesting because it's based on a topic I had previously done in class but never really got to use. In this post, I would give a brief explanation of the FFT algorithm stating its mathematical background. In the blogpost following this, various uses-cases of the FFT in Python would explored.


The FFT algorithm is an implementation of the Discrete Fourier Transform which is a type of Fourier Transform. The other type is the  __Inverse Discrete Fourier Transform__. It's an adaptation of the Fourier Transform, which in simple terms is an "attempt at digitizing the analog world".

### The Fourier Transform
Everything can be described using a waveform. Waveforms are a function of time or any other variable(space for an example). The Fourier Transform provides a rather useful way of looking at these waveforms. 
In its basic form, a Fourier Transform breaks down a waveform into sinusoids.This siusoids go a long way into proving that waves are not made of discrete number of frequencies but rather of a continuous range of frequencies. 

In mathematical terms, a Fourier Transform converts a time-based signal(wave, filter etc) into a sum of its sine and cosine waves with varying amplitude and wavelength.

For each frequency of wave in the signal, it assigns a complex-valued coefficient. This is called the Fourier coefficient.
The real part of this coefficient contains information about the waves' cosine amplitude and the imaginary part contains information about the waves' sine amplitude.


### The Discrete Fourier Transform
The DFT is further divided into two.


The Forward Transform represented as


$$F(m)  = {1 \over N}\sum_{n=0}^{N-1} f(n) \cdot e^{-i2 \pi mn \over N} $$  
      

The Backward/Inverse Transform represented as


$$f(n)  = \sum_{m=0}^{N-1} F(m) \cdot e^{-i2 \pi mn \over N} $$        


$$ f(n) $$ in both equations above is the value of the function $$ f $$ at point n. It can be either real or complex-valued.
$$ F(m) $$ on the other hand, is the coefficient for the $$ m^{th} $$ wave component and can only be complex. 


The movement from $$ f(n) $$ to $$ F(m) $$ defines a change in configuration from spacial to frequency based configurations. The Fourier coefficients at this point is represented as $$v_m  = {m \over t_s N} $$ where m=0, 1, 2, ..., $${N \over 2}$$    for positive frequencies and as $$v_m  = -{ (N - m + 1)\over t_s N} $$ where m=$${({N\over 2}) + 1}, {({N\over 2}) + 2}, {({N\over 2}) + 3}, ... , {({N\over 2})  + N}$$ for negative frequencies.



### Python Implementation

Given its relevance in so many areas, there are a lot of wrappers for computing the __DFT__ in Python. __Numpy__ has its ```numpy.fft``` function and __scipy__ has its ```scipy.fftpack``` implementation of this algorithm. 

Before looking at these use cases, let's attempt to write a dummy implementation of the __DFT__ in python.

```python 
 import math 
 import cmath
 import numpy
```


```python
def dft_with_complex_input(input_array):
    input_array_length, dft_matrix = len(input_array), [] 
    for number in range(input_array_length):  
        array_complex = complex(0)
        for num in range(input_array_length):
            angle = 2j * cmath.pi * num * number / input_array_length 
            array_complex += input_array[num] * cmath.exp(-angle)
        dft_matrix.append(array_complex)
    return dft_matrix
```

The above function iterates through each item in the input data and manually assigns them to fields in the __DFT__ formula above. 

>The input in this case has to be an array of real values.

Checking the result by comparing it to numpy's FFT implementation, we get:

```python
test_data = numpy.random.random(200)
numpy.allclose(dft_with_complex_input(test_data),numpy.fft.fft(test_data))
```
    True

Looks like everything is working fine

Timing the __dft_with_complex_input__ and the standard numpy implementation to view run_time differences, we get

```python
%timeit standard_dft_with_complex(test_data)
%timeit numpy.fft.fft(test_data)
```

```
    1 loop, best of 3: 136 ms per loop
    The slowest run took 16.43 times longer than the fastest. This could mean that an intermediate result is being cached.
    100000 loops, best of 3: 11.7 µs per loop

```

From the time difference above, the **__dft_with_complex_input__** function is more than ten thousand times slower than the numpy implementation.

Why?

Its easy. 

To start with, the `dft_with_complex_input` is a rather simple implementation involving loops.
Then it scales as an $$O[N^2]$$ whereas the standard numpy implementation scales as an  $$O[N log N]$$ .



An alternative implementation of the __DFT__ is shown below.

```python
def dft_with_input_pair(real_input, imaginary_input):
    if len(real_input) != len(imaginary_input):
        raise ValueError("The lengths should be equal")
	input_array_length, real_output, imaginary_output = len(inreal), [], []
    for num_iterator_one in range(input_array_length):
        real_sum, imaginary_sum = 0.0, 0.0
		for num_iterator_two in range(input_array_length):  # For each input element
			angle = 2 * math.pi * num_iterator_two * num_iterator_one / input_array_length
			real_sum +=  real_input[t] * math.cos(angle) + imaginary_input[t] * math.sin(angle)
			imaginary_sum += -real_input[t] * math.sin(angle) + imaginary_input[t] * math.cos(angle)
		real_output.append(real_sum)
		imaginary_output.append(imaginary_sum)
	return (outreal, outimag)
```
>The above function uses a similar approach as the first one but uses a pair of real and imaginary input.


A third implementation of the __DFT__ in Python is shown below.
It uses optimised numpy array functions.

```python
def dft_with_numpy(input_data):
    input_to_array = numpy.asarray(input_data, float)
    input_array_shape = input_to_array.shape()
    input_array_range = numpy.arange(input_array_shape)
    input_array_rearrange = numpy.reshape((input_array_shape, 1))
    output_matrix_vector = numpy.exp(-2j * numpy.pi * input_array_rearrange * input_array_range / input_array_shape)
    return numpy.dot(output_matrix_vector, input_to_array)
```

N.B : You can go ahead and check the output of each function on your own and also compare run times with that of the standard numpy and scipy implementations of the __DFT__.


### Scipy FFT
Scipy has an extensive support for the  Discrete Fourier Transform. It provides support for 3 types of FFTs.

1. Standard FFTs
2. Real FFTs
3. Hermitian FFTs


### Numpy FFT
Like scipy, numpy has a very thorough and very documented support for the FFT. It provides support for FFT with 5 functions under the `numpy.fft` module. These functions are:

1. numpy.fft.fft : This provides support for 1-Dimensional FFTs
2. numpy.ifft2 : This provides support for inverse 2-Dimensional FFTs
3. numpy.fftn : This provides support for n-D FFTs
4. numpy.fftshift: This shifts zero-frequency terms to the center of the array. For two-dimensional input, swaps first and third quadrants, and second and fourth quadrants.


If you are able to get to this point, you should have a clear idea of what the FFT algorithm is and how it can be implemented in various forms in Python.
In the accompanying post, I would look into each scipy and numpy fft function and give a detailed outline of their use-cases.