import torch.distributed as dist
import wandb
from datasets import load_dataset

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from oslo.torch.nn.parallel.tensor_parallel import TensorParallel
from oslo.torch.nn.parallel.utils import allocate_params
from oslo.torch.distributed import ParallelContext, ParallelMode
import time


# 병렬 사이즈 설정
tp_size = 4
tp_depth = 1

model_name = 'EleutherAI/polyglot-ko-1.3b'
dataset_name = "squad_kor_v1"

# parallel context 생성
parallel_context = ParallelContext.from_torch(
    data_parallel_size=1,
    pipeline_parallel_size=1,
    tensor_parallel_size=tp_size,
    tensor_parallel_mode=ParallelMode.TENSOR_1D,
    tensor_parallel_depth=1,
)


# 토크나이저 생성
tokenizer = AutoTokenizer.from_pretrained(
  model_name
)

# 모델 생성 및 병렬화 수행
model_tp = AutoModelForCausalLM.from_pretrained(
    model_name,
    pad_token_id=tokenizer.eos_token_id,
    torch_dtype='fp32', low_cpu_mem_usage=True
)
wrapper_tp = TensorParallel(model_tp, parallel_context)
allocate_params(wrapper_tp, parallel_context)


if dist.get_rank() == 0:
    print(wrapper_tp)

# 옵티마이저 생성
optimizer_tp = Adam(wrapper_tp.parameters(), lr=3e-5)

# 데이터셋 생성
batch_size = 4
datasets = load_dataset(dataset_name).data["train"]["context"]
datasets = [str(sample) for sample in datasets[:500]]
dataloader = DataLoader(datasets, batch_size=batch_size)

# 모니터링 생성
if dist.get_rank() == 0:
    wandb.init(project="oslo", name=f"{model_name}_tp2d_bs{batch_size}")
    cur = time.time()

# 모니터링 생성 대기
dist.barrier()

# 학습 시작
for data in dataloader:
    optimizer_tp.zero_grad()

    inputs = tokenizer(
        data,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to("cuda")

    loss_tp = wrapper_tp(**inputs, labels=inputs["input_ids"]).loss

    if dist.get_rank() == 0:
        print(f"loss:{loss_tp}")
        wandb.log({"loss": loss_tp})

    _ = loss_tp.backward()
    optimizer_tp.step()

# 저장
wrapper_tp.save_parallelized("test/", merge_checkpoints=False)
dist.barrier()
