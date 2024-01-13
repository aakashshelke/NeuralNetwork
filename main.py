import numpy as np
import argparse
import pandas as pd


def activation(n):
    return 1 / (1 + np.exp(-n))


def neural_net(data, eta, iterations):
    weight_a_h1 = -0.3
    weight_b_h1 = 0.4
    weight_bias_h1 = 0.2
    weight_a_h2 = -0.1
    weight_b_h2 = -0.4
    weight_bias_h2 = -0.5
    weight_a_h3 = 0.2
    weight_b_h3 = 0.1
    weight_bias_h3 = 0.3
    weight_h1_op = 0.1
    weight_h2_op = 0.3
    weight_h3_op = -0.4
    weight_bias_op = -0.1

    print(
        "a,b,h1,h2,h3,o,t,delta_h1,delta_h2,delta_h3,delta_o,w_bias_h1,w_a_h1,w_b_h1,w_bias_h2,w_a_h2,w_b_h2,w_bias_h3,w_a_h3,w_b_h3,w_bias_o,w_h1_o,w_h2_o,w_h3_o")
    print(
        f"-,-,-,-,-,-,-,-,-,-,-,{weight_bias_h1},{weight_a_h1},{weight_b_h1},{weight_bias_h2},{weight_a_h2},{weight_b_h2},{weight_bias_h3},{weight_a_h3},{weight_b_h3},{weight_bias_op},{weight_h1_op},{weight_h2_op},{weight_h3_op}")
    for _ in range(iterations):
        for datapoint in range(len(data)):
            h1_in = weight_bias_h1 * 1 + weight_a_h1 * data[datapoint][1] + weight_b_h1 * data[datapoint][2]
            h2_in = weight_bias_h2 * 1 + weight_a_h2 * data[datapoint][1] + weight_b_h2 * data[datapoint][2]
            h3_in = weight_bias_h3 * 1 + weight_a_h3 * data[datapoint][1] + weight_b_h3 * data[datapoint][2]

            h1_out = activation(h1_in)
            h2_out = activation(h2_in)
            h3_out = activation(h3_in)

            o_in = h1_out * weight_h1_op + h2_out * weight_h2_op + h3_out * weight_h3_op + weight_bias_op

            final_op = activation(o_in)

            error = data[datapoint][-1] - final_op
            delta_out = final_op * (1 - final_op) * error
            delta_h1 = h1_out * (1 - h1_out) * weight_h1_op * delta_out
            delta_h2 = h2_out * (1 - h2_out) * weight_h2_op * delta_out
            delta_h3 = h3_out * (1 - h3_out) * weight_h3_op * delta_out

            weight_h1_op += eta * delta_out * h1_out
            weight_h2_op += eta * delta_out * h2_out
            weight_h3_op += eta * delta_out * h3_out
            weight_bias_op += eta * delta_out

            weight_a_h1 += eta * delta_h1 * data[datapoint][1]
            weight_a_h2 += eta * delta_h2 * data[datapoint][1]
            weight_a_h3 += eta * delta_h3 * data[datapoint][1]

            weight_b_h1 += eta * delta_h1 * data[datapoint][2]
            weight_b_h2 += eta * delta_h2 * data[datapoint][2]
            weight_b_h3 += eta * delta_h3 * data[datapoint][2]

            weight_bias_h1 += eta * delta_h1
            weight_bias_h2 += eta * delta_h2
            weight_bias_h3 += eta * delta_h3

            print(data[datapoint][1],data[datapoint][2],h1_out,h2_out,h3_out,final_op,data[datapoint][3],delta_h1,delta_h2,delta_h3,delta_out,weight_bias_h1,weight_a_h1,weight_b_h1,weight_bias_h2,weight_a_h2,weight_b_h2,weight_bias_h3,weight_a_h3,weight_b_h3,weight_bias_op,weight_h1_op,weight_h2_op,weight_h3_op)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--eta", type=float, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    args = parser.parse_args()
    data_path = args.data
    data = pd.read_csv(data_path, header=None).values
    data = np.insert(data, 0, 1, axis=1)
    neural_net(data, args.eta, args.iterations)
