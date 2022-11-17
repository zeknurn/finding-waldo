import numpy
import pandas as pd

# Sum and avg. of non waldo glcm values.
df = pd.read_csv('probabilities_notwaldo.csv')
nr_of_values = numpy.count_nonzero(df)
sum = df.to_numpy().sum()
print('Not Waldo')
print(sum)
print(sum / nr_of_values)

# Sum and avg. of waldo glcm values.
df = pd.read_csv('probabilities_waldo.csv')
nr_of_values = numpy.count_nonzero(df)
sum = df.to_numpy().sum()
print('Waldo')
print(sum)
print(sum / nr_of_values)