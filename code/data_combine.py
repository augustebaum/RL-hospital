#! /usr/bin/env python3
import sys
import numpy as np

if __name__ == "__main__":

# Do I have to close it every time, even if I keep the same name?
    # Load the first file
    dict = np.load(str(sys.argv[1]))
    print(str(np.shape(dict['arr_0'])[0])+" data points")
    data = tuple(map(lambda x: x[1], dict.items()))
    dict.close()

    for i in range(2, len(sys.argv)):
        dict = np.load(str(sys.argv[i]))
        print(str(np.shape(dict['arr_0'])[0])+" data points")
        data = tuple(np.concatenate((a1, a2), axis=0) for (a1, a2) in zip(data, tuple(map(lambda x: x[1], dict.items()))))
        dict.close()

    print(np.shape(data[0]))
    np.savez("exp3/exp3-combined-50episodes100steps", *data)
