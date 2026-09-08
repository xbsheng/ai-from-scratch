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

    def compute_hash(self, token_ids: list[int], pre_hash=-1):
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
