#!/bin/bash

torchserve --start --model-store model_store --models trocr_base=trocr_base.mar --ts-config config.properties

read -p "TorchServe started. Press Enter to exit..."