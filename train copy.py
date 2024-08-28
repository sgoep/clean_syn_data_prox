from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import config
from src.data.data_loader import DataLoader
from src.models.unet import UNet

device = config.device


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(
            m.weight, mean=0.0, std=np.sqrt(2/(3**2 * m.in_channels))
        )
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=np.sqrt(2 / m.in_features))
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def train(
    model,
    TrainDataGenerator,
    TestDataGenerator,
    error,
    optimizer,
    device,
    NumEpochs,
    len_train,
    len_test
) -> Tuple[Dict, List, List]:

    torch.cuda.empty_cache()
    # gc.collect()

    history = {}
    history["loss"] = []
    history["val_loss"] = []
    train_loss_list = np.zeros(NumEpochs)
    val_loss_list = np.zeros(NumEpochs)

    for epoch in np.arange(1, NumEpochs+1):
        train_loss = 0.0
        val_loss = 0.0

        model.train()
        for X, Y in TrainDataGenerator:
            optimizer.zero_grad()

            X = X.to(device=device, dtype=torch.float)
            Y = Y.to(device=device, dtype=torch.float)

            output, _ = model(Y)

            loss = error(output.double(), X.double())

            loss.backward()
            optimizer.step()

            train_loss += loss.data.item()
            history["loss"].append(train_loss)

        model.eval()

        for X, Y in TrainDataGenerator:

            X = X.to(device=device, dtype=torch.float)
            Y = Y.to(device=device, dtype=torch.float)

            output, _ = model(Y)

            loss = error(output.double(), X.double())

            val_loss += loss.data.item()
            history["val_loss"].append(val_loss)

        train_loss = train_loss/len_train
        val_loss = val_loss/len_test

        print(
            f"Epoch {epoch}/{NumEpochs}, Train Loss: {train_loss:.8f} and "
            f"Val Loss: {val_loss:.8f}"
        )

        train_loss_list[epoch-1] = train_loss
        val_loss_list[epoch-1] = val_loss

    return history, train_loss_list, val_loss_list


def start_training(
    constraint: bool,
    initrecon: bool,
    null_space: bool,
    model_name: str,
    model_params: dict,
    training_params: dict,
    ell2_norm=None
) -> None:

    model = UNet(
        n_channels=1,
        n_classes=1,
        constraint=constraint,
        null_space=null_space,
        norm2=ell2_norm
    ).to(device)

    torch.manual_seed(42)
    model.apply(init_weights)

    model_optimizer = optim.Adam(
        model.parameters(), lr=training_params["learning_rate"]
    )
    model_error = nn.MSELoss()

    model_train_set = DataLoader(
        [i for i in range(training_params["len_train"])],
        constraint,
        initrecon,
        config.initial_regularization_method
    )

    model_test_set = DataLoader(
        [i for i in range(
            training_params["len_train"],
            training_params["len_train"] + training_params["len_test"]
        )],
        constraint,
        initrecon,
        config.initial_regularization_method
    )

    TrainDataGen = torch.utils.data.DataLoader(model_train_set, **model_params)
    TestDataGen = torch.utils.data.DataLoader(model_test_set, **model_params)

    print("##################### Start training ######################### ")
    print(f"Model: {model_name}")
    print(f"Size training set: {training_params['len_train']}")
    print(f"Size test set: {training_params['len_test']}")
    print(f"Epochs: {training_params['epochs']}")
    print(f"Learning rate: {training_params['learning_rate']}")
    print(f"Training on: {device}")
    print("############################################################## ")

    history, train_loss_list, val_loss_list = train(
        model,
        TrainDataGen,
        TestDataGen,
        model_error,
        model_optimizer,
        device,
        training_params["epochs"],
        training_params["len_train"],
        training_params["len_test"]
    )

    loss_plot = "losses/loss_" + model_name + ".pdf"

    plt.figure()
    plt.plot(train_loss_list,  marker="x", label="Training Loss")
    plt.plot(val_loss_list,  marker="x", label="Validation Loss")
    plt.ylabel("loss_" + model_name, fontsize=22)
    plt.legend()
    plt.savefig(loss_plot)

    print("######################## Saving model ######################## ")
    print("###### " + model_name + " ######") 
    torch.save(model.state_dict(), f"models/{model_name}")
    print("########################## Finished ########################## ")


if __name__ == "__main__":
    
    model_params = {"batch_size": 16,
                    "shuffle": True,
                    "num_workers": 2}
    training_params = {"learning_rate": config.learning_rate,
                       "len_train": config.len_train,
                       "len_test": config.len_test,
                       "epochs": config.epochs}


    # # FBP + RES
    # start_training(constraint = False, 
    #                initrecon = False,
    #                null_space = False,
    #                model_name = "fbp_resnet",
    #                model_params = model_params,
    #                training_params = training_params)
    
    # # L1 + RES
    # start_training(constraint = False,
    #                initrecon = True,
    #                null_space = False,
    #                model_name = f"{config.initial_regularization_method}_resnet", 
    #                model_params = model_params,
    #                training_params = training_params)
    
    # FBP + NSN
    start_training(constraint = False,
                   initrecon = False,
                   null_space = True,
                   model_name = "fbp_nsn",
                   model_params = model_params,
                   training_params = training_params)
    
    # L1 + NSN
    # start_training(constraint = False,
    #                initrecon = True,
    #                null_space = True,
    #                model_name = f"{config.initial_regularization_method}_nsn",
    #                model_params = model_params,
    #                training_params = training_params)

    ell2_norm = np.load("data/norm2.npy", allow_pickle=True)
    # ell2_norm *= 2
    # ell2_norm *= 0.5
    # ell2_norm *= 100
    ell2_norm *= config.factor
    # 10
    # MSE:  0.0006881785226412608
    # SSIM: 0.9431383774260057
    # PSNR: 31.256451533840885
    # 1
    # MSE:  0.0006702233982390131
    # SSIM: 0.9404519994795898
    # PSNR: 31.044065605928637
    # L1 + DP
    # start_training(constraint = True,
    #                initrecon = True, 
    #                null_space = True,
    #                model_name = f"{config.initial_regularization_method}_dpnsn",
    #                model_params = model_params,
    #                training_params = training_params,
    #                ell2_norm = ell2_norm)
    
