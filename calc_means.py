import numpy as np
import pandas as pd

if __name__ == "__main__":
    for teacher in ["exponential_decay", "step_decay", "sgdr", "constant"]:
        agg_run_data = pd.read_csv(f"data/SGD_CIFAR10_random_seed_20/SGD/{teacher}/0/aggregated_run_data.csv")

        final_evaluations = agg_run_data.groupby("run").last()
        train_loss = final_evaluations["train_loss"]
        valid_loss = final_evaluations["valid_loss"]
        test_loss = final_evaluations["test_loss"]
        train_acc = final_evaluations["train_acc"]
        valid_acc = final_evaluations["valid_acc"]
        test_acc = final_evaluations["test_acc"]
        print(f"{teacher}:")
        # print(f"Train Loss: {train_loss.mean()}")
        # print(f"Valid Loss: {valid_loss.mean()}")
        # print(f"Test Loss: {test_loss.mean()}")
        print(f"Train Acc: {train_acc.max()*100}% - {train_acc.min()*100}% = {(train_acc.max() - train_acc.min())*100}%")
        print(f"Valid Acc: {valid_acc.max()*100}% - {valid_acc.min()*100}% = {(valid_acc.max() - valid_acc.min())*100}%")
        print(f"Test Acc: {test_acc.max()*100}% - {test_acc.min()*100}% = {(test_acc.max() - test_acc.min())*100}%")
        print("\n")