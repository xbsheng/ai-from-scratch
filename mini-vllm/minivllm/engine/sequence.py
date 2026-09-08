from copy import copy
from enum import Enum, auto
from itertools import count

from minivllm.sampling_params import SamplingParams


class SeqStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    counter = count()
    block_size = 256

    def __init__(self, token_ids: list[int], sampling_params: SamplingParams):
        self.seq_id = next(self.counter)
        self.status = SeqStatus.WAITING

        self.is_prefill = True
        self.block_table = []

        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.num_scheduled_tokens = 0

        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, i: int):
        return self.token_ids[i]

    @property
    def is_finished(self):
        return self.status == SeqStatus.FINISHED

    @property
    def num_tokens(self):
        return len(self.token_ids)

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[: self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens :]

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def get_block_token_ids(self, i: int):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i * self.block_size : (i + 1) * self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
