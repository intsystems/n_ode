""" launches neural ode train processes for all types of activities
"""
import train
import yaml


def main():
    with open("../../data/dataset_params.yaml") as f2:
        data_params = yaml.full_load(f2)

    for activity in data_params["activity_codes"].keys():
        train.main(activity)


if __name__ == "__main__":
    main()