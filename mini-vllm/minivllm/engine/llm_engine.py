import atexit
from dataclasses import fields
from multiprocessing.context import SpawnProcess
from multiprocessing.synchronize import Event
from time import perf_counter

import torch.multiprocessing as mp
from minivllm.config import Config
from minivllm.engine.model_runner import ModelRunner
from minivllm.engine.scheduler import Scheduler
from minivllm.engine.sequence import Sequence
from minivllm.sampling_params import SamplingParams
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer


class LLMEngine:
    def __init__(self, model_name: str, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model_name, **config_kwargs)

        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)

        assert isinstance(self.tokenizer.eos_token_id, int)
        config.eos = self.tokenizer.eos_token_id

        Sequence.block_size = config.kv_cache_block_size

        (
            self.model_runner,
            self.ps,
            self.events,
        ) = self._init_model_runner(config)

        # scheduler 需要在 runner 初始化之后：
        # runner 初始化时会计算 num_kv_cache_blocks 并赋值到 config，scheduler 初始化时需要用到
        self.scheduler = Scheduler(config)

        # 注册一个"进程退出时自动执行"的回调
        # 当 Python 解释器正常结束时（比如主程序跑完、或调用 sys.exit()），atexit 模块会在退出前依次调用所有注册的函数
        # 进程正常退出时自动清理 TP 子进程和共享内存
        atexit.register(self.exit)

    def _init_model_runner(self, config: Config):
        ps: list[SpawnProcess] = []  # 保存所有 worker 进程句柄，exit() 时 join
        events: list[Event] = []  # 每个 worker 一个跨进程 Event，rank0 用来唤醒它们
        ctx = mp.get_context("spawn")  # 用 spawn 启动方式（新解释器，CUDA 安全）

        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()  # 给这个 worker 专属的信号量

            process = ctx.Process(target=ModelRunner, args=(config, i, event))

            # 子进程直接跑 ModelRunner 的 __init__
            # 对 rank>0 的 worker，__init__ 走到末尾会调用 self.loop()，进入一个永不返回的事件循环
            process.start()

            ps.append(process)
            events.append(event)

        # 主进程原地构造 rank 0 的 runner（不 spawn）
        # 这里传的是整个 events 列表，而 worker 传的是单个 event :
        # 因为 rank 0 是驱动方，write_shm 需要同时唤醒所有 worker
        model_runner = ModelRunner(config, 0, events)

        return model_runner, ps, events

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()
        print("exit success")

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        token_ids = self.tokenizer.encode(prompt) if isinstance(prompt, str) else prompt
        seq = Sequence(token_ids, sampling_params)
        self.scheduler.add_seq(seq)

    def is_finished(self):
        return self.scheduler.is_finished()

    def step(self) -> tuple[dict[int, list[int]], int, bool]:
        seqs, is_prefill = self.scheduler.schedule()
        num_step_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else len(seqs)

        token_ids = self.model_runner.run(seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)

        outputs = {seq.seq_id: seq.completion_token_ids for seq in seqs if seq.is_finished}

        return outputs, num_step_tokens, is_prefill

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm=True,
    ):
        progress_bar = tqdm(
            total=len(prompts),
            desc="Generating",
            dynamic_ncols=True,
            disable=not use_tqdm,
        )

        if isinstance(sampling_params, SamplingParams):
            sampling_params = [sampling_params] * len(prompts)

        for prompt, sampling_param in zip(prompts, sampling_params):
            self.add_request(prompt, sampling_param)

        outputs: dict[int, list[int]] = {}
        prefill_throughput = decode_throughput = 0
        while not self.is_finished():
            t = perf_counter()
            step_outputs, num_step_tokens, is_prefill = self.step()
            throughput = num_step_tokens // (perf_counter() - t)

            outputs.update(step_outputs)

            progress_bar.update(len(step_outputs))

            if is_prefill:
                prefill_throughput = throughput
            else:
                decode_throughput = throughput
            progress_bar.set_postfix(
                {
                    "Prefill": f"{prefill_throughput}tok/s",
                    "Decode": f"{decode_throughput}tok/s",
                }
            )

        progress_bar.close()

        return [
            {"text": self.tokenizer.decode(token_ids := outputs[seq_id]), "token_ids": token_ids}
            for seq_id in sorted(outputs)
        ]


if __name__ == "__main__":
    LLMEngine("Qwen/Qwen3-0.6B", max_num_batched_tokens=1000)
