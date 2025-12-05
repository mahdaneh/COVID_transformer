# agent.py
import subprocess


def run_agent():
    #
    print("======== resnet_vit")
    res_vit = subprocess.check_output(
        [
            "python3",
            "run_train_eval.py",
            "--config",
            "configs/Res_MyVIT_b_16.json",
            "--checkpoint_path",
            "weights/Res_MyVIT_b_16_1e-5_checkpoint_95.pth",
        ]
    )
    print("Resnet model")

    Resnet = subprocess.check_output(
        ["python3", "run_train_eval.py", "--config", "configs/Resnet.json"]
    )

    print("===== best_model_torch vision_vit_weightinitialize_from_ImageNet1K model")

    TorchVision_vit = subprocess.check_output(
        ["python3", "run_train_eval.py", "--config", "configs/TV_VIT_b_16.json"]
    )


if __name__ == "__main__":
    run_agent()
