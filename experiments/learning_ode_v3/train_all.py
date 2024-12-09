""" launches neural ode train processes for all types of activities
"""
import train
import yaml


def main():
    with open("../../data/dataset_params.yaml") as f2:
        data_params = yaml.full_load(f2)

    for activity, act_codes in data_params["activity_codes"].items():
        for act_code in act_codes:
            for partic_id in range(data_params["num_participants"]):
                print(f"Activity: {activity}; Act_code: {act_code}; Participant: {partic_id}")

                train.main(activity, act_code, partic_id)


if __name__ == "__main__":
    main()
