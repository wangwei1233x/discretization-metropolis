import numpy as np
from srcs.RMMetropolis import RMMetropolis
import argparse
import pandas as pd
import pickle
import os

##Set seeds
np.random.seed(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_iter", type=int, default=10000)
    parser.add_argument("--burn_in", type = int, default=5000)
    ## Multi-chain sampling will be supported in the future
    parser.add_argument("--num_chain", type=int, default=1)
    parser.add_argument("--total_words", type=int, default=40)
    parser.add_argument("--total_categories", type=int, default=7)
    parser.add_argument("--min_index", type=int, default=2)
    parser.add_argument("--max_index", type=int, default=39)
    parser.add_argument("--variance_hyper", type=float, default=4.0)
    parser.add_argument("--data_dir", type = str, default= "./data")
    config = parser.parse_args()

    #Setup random walk metropolis algorithm

    model = RMMetropolis(
        config.total_categories,
        config.total_words,
        config.min_index,
        config.max_index,
        config.burn_in,
        config.num_iter,
    )

    model.initialize()

    data_path = os.path.join(config.data_dir, "exp11_data.pkl")

    with open(data_path, 'rb') as file:
       data = pickle.load(file)

    # sample and analyze the result

    model.sample(
        data,
        config.variance_hyper
    )

    model.plotting()

    print(model.DIC(data))


