from multiprocessing.synchronize import Event

from minivllm.config import Config
from minivllm.engine.sequence import Sequence


class ModelRunner:
    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        pass

    def call(self, method_name: str):
        pass

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        return []
