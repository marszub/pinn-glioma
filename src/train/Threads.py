import asyncio
import contextlib
import sys
import termios

from train.Traininer import Trainer
from train.InteractionManager import InteractionManager

@contextlib.contextmanager
def raw_mode(file):
    old_attrs = termios.tcgetattr(file.fileno())
    new_attrs = old_attrs[:]
    new_attrs[3] = new_attrs[3] & ~(termios.ECHO | termios.ICANON)
    try:
        termios.tcsetattr(file.fileno(), termios.TCSADRAIN, new_attrs)
        yield
    finally:
        termios.tcsetattr(file.fileno(), termios.TCSADRAIN, old_attrs)

async def keyReaderThread(interactionManager: InteractionManager):
    with raw_mode(sys.stdin):
        reader = asyncio.StreamReader()
        loop = asyncio.get_event_loop()
        await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), sys.stdin)

        while not reader.at_eof():
            ch = await reader.read(1)
            # '' means EOF, chr(4) means EOT (sent by CTRL+D on UNIX terminals)
            if not ch or ord(ch) <= 4:
                break
            interactionManager.dispatchInput(ch)

async def interactiveTrainingThread(trainer: Trainer, interactionManager: InteractionManager):
    await asyncio.sleep(0)
    keyReaderTask = asyncio.create_task(keyReaderThread(interactionManager))
    await trainer.train()
    keyReaderTask.cancel()

async def trainingThread(trainer: Trainer):
    await trainer.train()
