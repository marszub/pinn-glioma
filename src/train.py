#!/bin/python


if __name__ == "__main__":
    from train.ArgsParser import ArgsParser

    argsParser = ArgsParser()
    argsParser.show()
    args = argsParser.get()

    import asyncio
    import torch
    from train.Initializer import Initializer
    from pinn.PinnConfig import PinnConfig
    from model.Experiment import Experiment
    from train.Traininer import Trainer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    experiment = Experiment()
    config = PinnConfig(experiment, args.data_dir, args.validation_dir)
    initializer = Initializer(args, config, device)

    trainer = Trainer(initializer, config.loss.to(device))
    asyncio.run(initializer.getMainThread(trainer))
