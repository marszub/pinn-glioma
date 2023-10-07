from train.tracking.SharedData import SharedData


class InteractionManager:
    def __init__(self, sharedData: SharedData):
        self.__sharedData = sharedData

    def dispatchInput(self, input: bytes):
        if input == b'e':
            self.__sharedData.save = True
            self.__sharedData.terminate = True
        elif input == b's':
            self.__sharedData.save = True
        else:
            self.__sharedData.unsupportedInput = True
