# own implementation of a "calculating" gpt

in the end, it should add two number by implementing the format:

f"{number1:05d}+{number2:05d}={number3:6d}"

for training, it should only see number up to 10**4. Also, certain primes shall be excluded to test the intrapolation capabilities.
For extrapolation capabilites, the numbers >10**4 will be tested

purpose:

1. write gpt from scratch for training myself
2. make experiments on generalization of gpts
