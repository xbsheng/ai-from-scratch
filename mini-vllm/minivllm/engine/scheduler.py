from collections import deque

from minivllm.config import Config
from minivllm.engine.block_manager import BlockManager
from minivllm.engine.sequence import SeqStatus, Sequence


class Scheduler:
    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos

        self.block_size = config.kv_cache_block_size
        self.block_manager = BlockManager(config.num_kv_cache_blocks, config.kv_cache_block_size)

        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def add_seq(self, seq: Sequence):
        self.waiting.append(seq)

    def is_finished(self):
        return not self.waiting and not self.running

    def schedule(self) -> tuple[list[Sequence], bool]:
        """prefill / decode

        Returns:
            tuple[list[Sequence], bool]: (scheduled_seqs, is_prefill)
        """
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]

            num_remaining_tokens = self.max_num_batched_tokens - num_batched_tokens
            if not num_remaining_tokens:
                break

            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_new_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_new_tokens = seq.num_tokens - seq.num_cached_tokens

            if num_remaining_tokens < num_new_tokens and scheduled_seqs:
                # 只有第一个 seq 可以部分 prefill (chunked prefill):
                #
                # 队头这个的 num_tokens 可能大于 max_num_batched_tokens，若不切，它永远进不了 batch，后面也堵死
                # 所以第一个（本轮 scheduled_seqs 还空时）即使额度不够也要 min(num_tokens, remaining) 做一段
                #
                # 已经排进几条完整 prefill 后，若剩余 token 不够下一条整段，代码直接 break，
                # 宁可这步浪费一点 budget，也不开第二条的部分 prefill
                break

            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)  # pyright: ignore[reportPossiblyUnboundVariable]

            # 只有第一个 seq 可能 num_remaining_tokens < num_new_tokens
            seq.num_scheduled_tokens = min(num_remaining_tokens, num_new_tokens)
            num_batched_tokens += seq.num_scheduled_tokens

            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SeqStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)

            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()

            while not self.block_manager.can_append(seq):
                # free block 不够时
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)

        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SeqStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0

            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                # chunked prefill
                continue

            seq.append_token(token_id)

            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SeqStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
