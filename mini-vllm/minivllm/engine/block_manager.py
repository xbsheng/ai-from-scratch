from collections import deque

import numpy as np
import xxhash
from minivllm.engine.sequence import Sequence


class Block:
    def __init__(self, block_id: int):
        self.block_id = block_id
        self.hash = -1
        self.ref_count = 0
        self.token_ids = []

    def update(self, h: int, token_ids: list[int]):
        self.hash = h
        self.token_ids = token_ids

    def reset(self):
        self.hash = -1
        self.ref_count = 1
        self.token_ids = []


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.hash_to_block: dict[int, int] = {}

        self.free_block_ids: deque[int] = deque()
        self.used_block_ids: set[int] = set()

    def compute_hash(self, token_ids: list[int], pre_hash=-1) -> int:
        """计算block hash

        Args:
            token_ids (list[int]): 当前block token_ids
            pre_hash (int, optional): 前一个block hash. Defaults to -1.

        Returns:
            hash (int): block hash
        """
        h = xxhash.xxh64()
        if pre_hash != -1:
            h.update(pre_hash.to_bytes(8, byteorder="little"))

        h.update(np.array(token_ids).tobytes())

        return h.intdigest()

    def _allocate_block(self):
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0

        if block.hash != -1 and self.hash_to_block[block.hash] == block.block_id:
            # 因为_deallocate_block操作中，向free追加的block没有清空hash
            del self.hash_to_block[block.hash]

        block.reset()
        self.used_block_ids.add(block_id)

        return block_id

    def _deallocate_block(self, block_id: int):
        block = self.blocks[block_id]
        assert block.ref_count == 0

        # 故意不清 hash / token_ids, free 里的 block 还能当 prefix cache：
        # 后续通过 hash 命中后，若 block 还在 free，就从 free 挪回 used
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence):
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks

        # why seq.num_blocks - 1 ?
        # 因为 最后一个 block 通常不满，不会进 prefix cache
        for i in range(seq.num_blocks - 1):
            block_token_ids = seq.get_block_token_ids(i)
            h = self.compute_hash(block_token_ids, h)
            cached_block_id = self.hash_to_block.get(h, -1)

            # 为什么添加 token_ids != block_token_ids 这个判断条件？
            # hash 相同但 token_ids 不同是哈希碰撞，概率极低，但仍可能发生
            # 不是为了常走，是为了万一撞了也不错用缓存
            if cached_block_id == -1 or self.blocks[cached_block_id].token_ids != block_token_ids:
                break

            num_cached_blocks += 1

            if cached_block_id in self.used_block_ids:
                num_new_blocks -= 1

        if len(self.free_block_ids) < num_new_blocks:
            return -1

        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        assert not seq.block_table

        h = -1

        # cached blocks
        for i in range(num_cached_blocks):
            token_ids = seq.get_block_token_ids(i)
            h = self.compute_hash(token_ids, h)

            block_id = self.hash_to_block[h]
            block = self.blocks[block_id]

            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)

            seq.block_table.append(block_id)

        # remaining blocks (no prefix cache hit)
        for i in range(num_cached_blocks, seq.num_blocks):
            block_id = self._allocate_block()
            seq.block_table.append(block_id)

        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence):
        # block_table 倒序遍历？
        # free_block_ids 是 deque：释放时 append 到队尾，分配时 popleft 从队头取，相当于 FIFO
        # 前缀更常被后续请求 hash 命中，留久一点，free 里带着 hash 的 block 更容易被复用，而不是刚释放就被拿去写新内容
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)

        seq.block_table.clear()
        seq.num_cached_tokens = 0

    def can_append(self, seq: Sequence):
        return len(self.free_block_ids) >= (seq.num_tokens % self.block_size == 1)

    def may_append(self, seq: Sequence):
        if seq.num_tokens % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence):
        # 新增block的起始、结尾索引值
        start_idx = seq.num_cached_tokens // self.block_size
        end_idx = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size

        if start_idx == end_idx:
            return

        h = self.blocks[seq.block_table[start_idx - 1]].hash if start_idx > 0 else -1

        for i in range(start_idx, end_idx):
            block_id = seq.block_table[i]
            block = self.blocks[block_id]

            token_ids = seq.get_block_token_ids(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block[h] = block_id
