import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'common')))
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import time
from formatter import Formatter
from probability_provider import ProbabilityProvider
from data_loader import DataLoader, Set
from logger import FileLogger
from model import GPT, GPTConfig
from utils import GenerationTools
from character_tokenizer import CharacterTokenizer
from generators.fixed_sums import FixedSums
from generators.random_sums import RandomSums
from torch.utils.tensorboard import SummaryWriter
from lr_scheduler import AdaptiveLearningRateScheduler

save_id = 1
B = 4*1024 # 4*1024 # micro batch size
T = 32 # sequence length
total_batch_size =8 * B*T # usually size of dataset or it's chunk, not applicable here
seed = 1337
CHAR_VOCAB = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '=', '\n', '_', 'X', 'Y']

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 100
max_steps = 10000 # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens


s = [
    Set(1000,FixedSums(1,9)),
    Set(1000,FixedSums(9,1)),
    Set(1000,FixedSums(1,8)),
    Set(1000,FixedSums(8,1)),
    Set(1000,FixedSums(8,2)),
    Set(1000,FixedSums(2,8)),
    Set(1000,FixedSums(7,3)),
    Set(1000,FixedSums(3,7)),
    Set(1000,FixedSums(5,5)),
    Set(1000,RandomSums(8)),
    Set(1000,RandomSums(10)),
    Set(1000,RandomSums(12)),
    Set(1000,RandomSums(15)),
    Set(1000,RandomSums(18)),
]

s2 = [
    Set(1000,FixedSums(1,9)),
    Set(1000,FixedSums(9,1)),
    Set(1000,FixedSums(1,8)),
    Set(1000,FixedSums(8,1)),
    Set(1000,FixedSums(8,2)),
    Set(1000,FixedSums(2,8)),
    Set(1000,FixedSums(7,3)),
    Set(1000,FixedSums(3,7)),
    Set(1000,FixedSums(5,5)),
    Set(1000,RandomSums(8)),
    Set(1000,RandomSums(10)),
    Set(1000,RandomSums(12)),
    Set(1000,RandomSums(15)),
    Set(1000,RandomSums(18)),
]

level_1_p = [0]*len(s)
level_1_p[0] = 1
level_1_p[1] = 1
level_1_p[2] = 0

def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)



tokenizer = CharacterTokenizer(CHAR_VOCAB)
probabilities = ProbabilityProvider(30, len(s))
probabilities.set_probabilities(level_1_p)
formatter = Formatter(tokenizer)
logger = FileLogger("logs")
tensorBoard = SummaryWriter()

train_loader = DataLoader(B, T, probabilities, formatter, sets = s)
val_loader = DataLoader(B, T, probabilities, formatter, sets = s2)

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
print(f'ddp={ddp}')
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

tools = GenerationTools(device)

# added after video, pytorch can be serious about it's device vs. device_type distinction
device_type = "cuda" if device.startswith("cuda") else "cpu"

if seed>-1:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

torch.set_float32_matmul_precision('high')


# create model
model = GPT(GPTConfig(vocab_size=formatter.tokenizer.get_vocab_size(), n_embd=64))
start_step = model.load(save_id, torch.device(device) )+1
scheduler = AdaptiveLearningRateScheduler(start_step ,max_lr, min_lr, warmup_steps, max_steps, 0.9, 0.999);
# model = GPT.from_pretrained("gpt2") # or init from OpenAI GPT-2
model.to(device)
use_compile = True 
if use_compile:
    model = torch.compile(model)
    print("model compiled")
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

# optimize!
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type)

correct_answers_threshold = 0.9

for step in range(start_step, max_steps):
    
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # once in a while evaluate our validation loss
    if step % 20 == 0 or last_step:
        model.eval()
        with torch.no_grad():
            val_loss_accum = 0.0
            x, y, samples = val_loader.next_batch()
            queries = [example.split('=')[0] + '=' for example in samples]
            answers, probs = tools.generate_answers(model, formatter, B, T, queries)
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            val_loss_accum += loss.detach()

        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)

        # Calculate the percentage of correct answers
        match_count, miss_count = GenerationTools.count_score(samples, answers)
        total_answers = match_count + miss_count
        correct_percentage = match_count / total_answers if total_answers > 0 else 0

        if step % 100 == 0 or last_step:
            model.save(step)

        if master_process:
            logger.log_info(f"validation loss: {val_loss_accum.item():.4f}")
            logger.log_info(f"Correct answers percentage: {correct_percentage * 100:.2f}%")
            tensorBoard.add_scalar("validation loss", val_loss_accum, step)
            tensorBoard.add_scalar("correctness", correct_percentage * 100, step)
            print(f"Correct answers percentage: {correct_percentage * 100:.2f}%")
            if correct_percentage > correct_answers_threshold:
                model.save(step)
                print("Time for challenge")
                probabilities.make_harder()
            logger.log_info(f"{step} val {val_loss_accum.item():.4f}\n")
        train_loader.reset()
        val_loader.reset()
        print(probabilities.get_probabilities())

    # do one step of the optimization
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y, sample = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        # added after video, this field is also used by the forward pass.
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        #print(f'logits.size()={logits.size()}')
        #print(f'y[0,1]={y[0,1]}')
        #print(f'logits[0,1,:]={logits[0,1,:]}')
        #print(f'y[0,14]={y[0,13]}')
        #print(f'logits[0,14,:]={logits[0,13,:]}')
        #print(f'y[0,14]={y[0,14]}')
        #print(f'logits[0,14,:]={logits[0,14,:]}')
        #print(f'y[0,15]={y[0,15]}')        
        #print(f'logits[0,15,:]={logits[0,15,:]}')

        # we have to scale the loss to account for gradient accumulation,
        # because the gradients just add on each successive backward().
        # addition of gradients corresponds to a SUM in the objective, but
        # instead of a SUM we want MEAN. Scale the loss here so it comes out right
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # determine and set the learning rate for this iteration
    lr = scheduler.get_lr()
    tensorBoard.add_scalar("lr", lr, step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    if device_type == "cuda":
        torch.cuda.synchronize() # wait for the GPU to finish work
    t1 = time.time()
    dt = t1 - t0 # time difference in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    scheduler.inform(loss_accum, norm)
    scheduler.next_step()
    if master_process:
        tensorBoard.add_scalar("loss", loss_accum, step)
        tensorBoard.add_scalar("norm", norm, step)
        logger.log_info(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")

if ddp:
    destroy_process_group()