# agent.py
import subprocess
import json


def run_agent():

    #
    # print("======== resnet_vit")
    # res_vit = subprocess.check_output(
    #     ["python3", "run_train_eval.py", "--config", "configs/Res_VIT_small.json"]
    # )

    print("vit model")
    vit = subprocess.check_output(
        ["python3", "run_train_eval.py", "--config", "configs/VIT_QU_EX_small.json"]
    )

if __name__ == "__main__":
    run_agent()
