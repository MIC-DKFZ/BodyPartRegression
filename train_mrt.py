import os
import argparse

from bpreg.scripts.train import train_json


if __name__=="__main__":
    base_dir = "/media/lisa/HDD/Data/examples/"
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", help="base directory containing preprocessed npy folder")
    args = parser.parse_args()

    train_json(os.path.join(args.input_dir, "bpr-config.json"))