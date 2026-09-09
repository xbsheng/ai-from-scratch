from minivllm.config import Config
from minivllm.engine.sequence import Sequence


class ModelRunner:
    def __init__(self, config: Config):
        pass

    def call(self, arg):
        pass

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        return []
