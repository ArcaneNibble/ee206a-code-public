#!/usr/bin/env python3

import numpy as np

layers = (
    [np.array([[1., 0], [0, 0]]), np.array([[0.], [0]])],
    [np.array([[1., 0], [0, 0]]), np.array([[0.], [0]])],
)
print(layers)

training = (
    (np.array([[1],[0]]), np.array(([0], [1]))),
    (np.array([[2],[0]]), np.array(([0], [2]))),
    (np.array([[0],[1]]), np.array(([1], [0]))),
    (np.array([[0],[2]]), np.array(([2], [0]))),
    (np.array([[1],[2]]), np.array(([2], [1]))),
    (np.array([[2],[1]]), np.array(([1], [2]))),
)
print(training)

for iter_ in range(10000):
    for inin, refref in training:
        learn_rate = 0.001

        # Forward
        netsum = []
        output = []
        for i in range(len(layers)):
            if i == 0:
                in_i = inin
            else:
                in_i = output[i - 1]

            netsum_i = layers[i][0] @ in_i + layers[i][1]
            # print("netsum", netsum_i)

            # relu
            output_i = np.clip(netsum_i, 0, None)
            # print("output", output_i)

            netsum.append(netsum_i)
            output.append(output_i)

        # Backwards
        errerr = output[-1] - refref
        err_scalar = (0.5 * errerr.T @ errerr)[0][0]
        print("err_scalar", err_scalar)

        errors = [None] * len(layers)
        for i in reversed(range(len(layers))):
            if i == len(layers) - 1:
                err_i_after_relu = errerr
            else:
                # print("AAAAA")
                err_i_1 = errors[i + 1]

                err_i_after_relu = layers[i + 1][0].T @ err_i_1
                # print(err_i_after_relu)

            relu_deriv = np.clip(np.sign(netsum[i]), 0, 1)
            # print(relu_deriv)

            err_i = err_i_after_relu * relu_deriv

            # print("err", err_i)

            errors[i] = err_i

        # print(errors)

        # Update weights
        for i in range(len(layers)):
            if i == 0:
                in_i = inin
            else:
                in_i = output[i - 1]

            err_activations = errors[i]
            # print("err_act", err_activations)
            err_bias = errors[i]
            # print("err_bias", err_bias)

            layers[i][0] -= learn_rate * err_activations
            layers[i][1] -= learn_rate * err_bias

        # print(layers)

print(layers)
