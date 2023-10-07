from train.tracking.SharedData import SharedData


class InteractionManager:
    def __init__(self, sharedData: SharedData):
        self.__sharedData = sharedData

    def dispatchInput(self, input: bytes):
        if input == b'e':
            print("After this epoch will save model and exit")
            self.__sharedData.save = True
            self.__sharedData.terminate = True
        elif input == b's':
            print("After this epoch will save model")
            self.__sharedData.save = True
        else:
            print("s      save model\ne      exit with saving\nCtrl+C exit without saving")
