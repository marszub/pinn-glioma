import os
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from tracking.Tracker import Tracker


class PauseMenu:
    def __init__(self, tracker: "Tracker"):
        self.tracker = tracker
        self.isTerminated = False
        self.isUnpaused = False
        self.commands = {
            "animate",
            "cd",
            "exit",
            "help",
            "mkdir",
            "resume",
        }

    def run(self) -> None:
        try:
            print()
            while not self.isUnpaused:
                try:
                    command = input(f"PINN:{self.tracker.visualizer.saveDir}> ")
                    self.__dispatch(command)
                except WrongCommandSyntax:
                    print(
                        "Wrong syntax in entered command. Type 'help' for more information."
                    )
        except KeyboardInterrupt:
            self.isTerminated = True

    def shouldTerminate(self):
        return self.isTerminated

    def __dispatch(self, command: str) -> None:
        command = command.split(" ")
        args = command[1:]
        command = command[0]
        if command not in self.commands:
            raise WrongCommandSyntax
        func = getattr(self, command)
        func(args)

    def animate(self, _):
        self.tracker.visualizer.animateProgress(self.tracker.bestApprox.cpu(), "animation")

    def cd(self, args):
        if len(args) != 1:
            print(f"cd requires 1 argument. {len(args)} were provided.")
            return
        path = args[0]
        if os.path.isdir(path):
            raise NotImplemented("Changing dir in visualizer requires designated method.")
        else:
            print(
                f"Directory '{path}' does not exist. To create it use 'mkdir'"
            )

    def exit(self, _):
        self.isTerminated = True
        self.isUnpaused = True

    def help(self, _):
        print("Supported commands: ")
        print(", ".join(self.commands))

    def mkdir(self, args):
        if len(args) != 1:
            print(f"mkdir requires 1 argument. {len(args)} were provided.")
            return
        path = args[0]
        if os.path.isfile(path):
            print(
                f"'{path}' is a file. Delete it or choose different dir name."
            )
            return
        os.makedirs(path, exist_ok=True)
        raise NotImplemented("Changing dir in visualizer requires designated method.")

    def resume(self, _):
        self.isUnpaused = True


class WrongCommandSyntax(Exception):
    pass
