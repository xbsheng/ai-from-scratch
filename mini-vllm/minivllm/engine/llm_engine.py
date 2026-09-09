from dataclasses import fields
from time import perf_counter

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
        self.config = Config(model_name, **config_kwargs)

        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)
        self.scheduler = Scheduler(self.config)
        self.model_runner = ModelRunner(self.config)

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
