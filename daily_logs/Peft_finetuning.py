# Make a 7B model memorise 50 examples with loss less than 0.5, this is about plumbing
import random
import wandb
import torch
from datasets import load_datasets
from trl import SFTTrainer
from transformers import TrainingArguments

# Initialize weights and biases for experiment tracking
wandb.init(project="pb1-overfit", name=f"run-{random.randint(0,999)}")
# Load a small subset of the dataset (50 examples)
dataset = load_dataset("codefuse-ai/CodeExercise-Python-27k", split="train[:50]")

# Format the dataset to create instruction-response pairs
def format_example(example):
    return {
        "text": f"### Question: {example['chat_rounds'][0]['content']}\n### Answer: {example['chat_rounds'][1]['content']}"
    }
dataset = dataset.map(format_example)

# define training arguments
training_args = TrainingArguments(
    output_dir = "ov",
    max_steps=100,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    learining_rate=2e-4,
    bf16=True,
    logging_steps=5,
    report_to="wandb"
)

# Initialize the trainer
trainer = SFTTrainer(
    model="meta-llama/Llama-2-7b-chat-hf",
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=256,
    args=training_args
)

# Start training
trainer.train()
# Make a 7B model memorise 50 examples with loss less than 50%, this is about plumbing
import random
import wandb
import torch
from datasets import load_datasets
from trl import SFTTrainer
from transformers import TrainingArguments

# initialize weights and biases for experiment tracking
wandb.init(projects="pb1-overfit", name=f"run-{random.randit(0,999)}")
# Load a small subset of the dataset (50 examples)
dataset= load_dataset("codefuse-ai/CodeExercise-Python-27k", split="train[:50]")

#Format the dataset to create instruction-response pairs
def format_example(example):
  return {
      "text": f"### Question: {example['chat_rounds'][0]['content']}\n### Answer: {example['chat_rounds'][1]['content']}"
  }
  dataset = dataset.map(format_example)
  # define training arguments
  training_args = TrainingArguments(
      output_dir = "ov",
      max_steps = 100,
      per_device_train_batch_size = 2,
      gradient_accumulation_steps = 1,
      learning_rate = 2e-4,
      bf16 = True,
      logging_steps = 5,
      report_to = "wandb"
  )
  # initialize the trainer
  trainer = SFTTrainer(
      model="meta-llama/Llama-2-7b-chat-hf",
      train_dataset=dataset,
      dataset_text_field = "text",
      max_seq_length = 256,
      args = training_args
  )
  # start training
  trainer.train()
  # Make a 7B model memorise 50 examples with loss less than 50%, this is about plumbing
import random
import wandb
import torch
from datasets import load_datasets
from trl import SFTTrainer
from transformers import TrainingArguments

# initialize weights and biases for experiment tracking
wandb.init(projects="pb1-overfit", name=f"run-{random.randit(0,999)}")
# Load a small subset of the dataset (50 examples)
dataset= load_dataset("codefuse-ai/CodeExercise-Python-27k", split="train[:50]")

#Format the dataset to create instruction-response pairs
def format_example(example):
  return {
      "text": f"### Question: {example['chat_rounds'][0]['content']}\n### Answer: {example['chat_rounds'][1]['content']}"
  }
  dataset = dataset.map(format_example)
  # define training arguments
  training_args = TrainingArguments(
      output_dir = "ov",
      max_steps = 100,
      per_device_train_batch_size = 2,
      gradient_accumulation_steps = 1,
      learning_rate = 2e-4,
      bf16 = True,
      logging_steps = 5,
      report_to = "wandb"
  )
  # initialize the trainer
  trainer = SFTTrainer(
      model="meta-llama/Llama-2-7b-chat-hf",
      train_dataset=dataset,
      dataset_text_field = "text",
      max_seq_length = 256,
      args = training_args
  )
  # start training
  trainer.train()
  # Make a 7B model memorise 50 examples with loss less than 50%, this is about plumbing
import random
import wandb
import torch
from datasets import load_datasets
from trl import SFTTrainer
from transformers import TrainingArguments

# initialize weights and biases for experiment tracking
wandb.init(projects="pb1-overfit", name=f"run-{random.randit(0,999)}")
# Load a small subset of the dataset (50 examples)
dataset= load_dataset("codefuse-ai/CodeExercise-Python-27k", split="train[:50]")

#Format the dataset to create instruction-response pairs
def format_example(example):
  return {
      "text": f"### Question: {example['chat_rounds'][0]['content']}\n### Answer: {example['chat_rounds'][1]['content']}"
  }
  dataset = dataset.map(format_example)
  # define training arguments
  training_args = TrainingArguments(
      output_dir = "ov",
      max_steps = 100,
      per_device_train_batch_size = 2,
      gradient_accumulation_steps = 1,
      learning_rate = 2e-4,
      bf16 = True,
      logging_steps = 5,
      report_to = "wandb"
  )
  # initialize the trainer
  trainer = SFTTrainer(
      model="meta-llama/Llama-2-7b-chat-hf",
      train_dataset=dataset,
      dataset_text_field = "text",
      max_seq_length = 256,
      args = training_args
  )
  # start training
  trainer.train()
  # Make a 7B model memorise 50 examples with loss less than 50%, this is about plumbing
import random
import wandb
import torch
from datasets import load_datasets
from trl import SFTTrainer
from transformers import TrainingArguments

# initialize weights and biases for experiment tracking
wandb.init(projects="pb1-overfit", name=f"run-{random.randit(0,999)}")
# Load a small subset of the dataset (50 examples)
dataset= load_dataset("codefuse-ai/CodeExercise-Python-27k", split="train[:50]")

#Format the dataset to create instruction-response pairs
def format_example(example):
  return {
      "text": f"### Question: {example['chat_rounds'][0]['content']}\n### Answer: {example['chat_rounds'][1]['content']}"
  }
  dataset = dataset.map(format_example)
  # define training arguments
  training_args = TrainingArguments(
      output_dir = "ov",
      max_steps = 100,
      per_device_train_batch_size = 2,
      gradient_accumulation_steps = 1,
      learning_rate = 2e-4,
      bf16 = True,
      logging_steps = 5,
      report_to = "wandb"
  )
  # initialize the trainer
  trainer = SFTTrainer(
      model="meta-llama/Llama-2-7b-chat-hf",
      train_dataset=dataset,
      dataset_text_field = "text",
      max_seq_length = 256,
      args = training_args
  )
  # start training
  trainer.train()
  # Make a 7B model memorise 50 examples with loss less than 50%, this is about plumbing
import random
import wandb
import torch
from datasets import load_datasets
from trl import SFTTrainer
from transformers import TrainingArguments

# initialize weights and biases for experiment tracking
wandb.init(projects="pb1-overfit", name=f"run-{random.randit(0,999)}")
# Load a small subset of the dataset (50 examples)
dataset= load_dataset("codefuse-ai/CodeExercise-Python-27k", split="train[:50]")

#Format the dataset to create instruction-response pairs
def format_example(example):
  return {
      "text": f"### Question: {example['chat_rounds'][0]['content']}\n### Answer: {example['chat_rounds'][1]['content']}"
  }
  dataset = dataset.map(format_example)
  # define training arguments
  training_args = TrainingArguments(
      output_dir = "ov",
      max_steps = 100,
      per_device_train_batch_size = 2,
      gradient_accumulation_steps = 1,
      learning_rate = 2e-4,
      bf16 = True,
      logging_steps = 5,
      report_to = "wandb"
  )
  # initialize the trainer
  trainer = SFTTrainer(
      model="meta-llama/Llama-2-7b-chat-hf",
      train_dataset=dataset,
      dataset_text_field = "text",
      max_seq_length = 256,
      args = training_args
  )
  # start training
  trainer.train()
  # Make a 7B model memorise 50 examples with loss less than 50%, this is about plumbing
import random
import wandb
import torch
from datasets import load_datasets
from trl import SFTTrainer
from transformers import TrainingArguments

# initialize weights and biases for experiment tracking
wandb.init(projects="pb1-overfit", name=f"run-{random.randit(0,999)}")
# Load a small subset of the dataset (50 examples)
dataset= load_dataset("codefuse-ai/CodeExercise-Python-27k", split="train[:50]")

#Format the dataset to create instruction-response pairs
def format_example(example):
  return {
      "text": f"### Question: {example['chat_rounds'][0]['content']}\n### Answer: {example['chat_rounds'][1]['content']}"
  }
  dataset = dataset.map(format_example)
  # define training arguments
  training_args = TrainingArguments(
      output_dir = "ov",
      max_steps = 100,
      per_device_train_batch_size = 2,
      gradient_accumulation_steps = 1,
      learning_rate = 2e-4,
      bf16 = True,
      logging_steps = 5,
      report_to = "wandb"
  )
  # initialize the trainer
  trainer = SFTTrainer(
      model="meta-llama/Llama-2-7b-chat-hf",
      train_dataset=dataset,
      dataset_text_field = "text",
      max_seq_length = 256,
      args = training_args
  )
  # start training
  trainer.train()
  # Make a 7B model memorise 50 examples with loss less than 50%, this is about plumbing
import random
import wandb
import torch
from datasets import load_datasets
from trl import SFTTrainer
from transformers import TrainingArguments

# initialize weights and biases for experiment tracking
wandb.init(projects="pb1-overfit", name=f"run-{random.randit(0,999)}")
# Load a small subset of the dataset (50 examples)
dataset= load_dataset("codefuse-ai/CodeExercise-Python-27k", split="train[:50]")

#Format the dataset to create instruction-response pairs
def format_example(example):
  return {
      "text": f"### Question: {example['chat_rounds'][0]['content']}\n### Answer: {example['chat_rounds'][1]['content']}"
  }
  dataset = dataset.map(format_example)
  # define training arguments
  training_args = TrainingArguments(
      output_dir = "ov",
      max_steps = 100,
      per_device_train_batch_size = 2,
      gradient_accumulation_steps = 1,
      learning_rate = 2e-4,
      bf16 = True,
      logging_steps = 5,
      report_to = "wandb"
  )
  # initialize the trainer
  trainer = SFTTrainer(
      model="meta-llama/Llama-2-7b-chat-hf",
      train_dataset=dataset,
      dataset_text_field = "text",
      max_seq_length = 256,
      args = training_args
  )
  # start training
  trainer.train()
  # Make a 7B model memorise 50 examples with loss less than 50%, this is about plumbing
import random
import wandb
import torch
from datasets import load_datasets
from trl import SFTTrainer
from transformers import TrainingArguments

# initialize weights and biases for experiment tracking
wandb.init(projects="pb1-overfit", name=f"run-{random.randit(0,999)}")
# Load a small subset of the dataset (50 examples)
dataset= load_dataset("codefuse-ai/CodeExercise-Python-27k", split="train[:50]")

#Format the dataset to create instruction-response pairs
def format_example(example):
  return {
      "text": f"### Question: {example['chat_rounds'][0]['content']}\n### Answer: {example['chat_rounds'][1]['content']}"
  }
  dataset = dataset.map(format_example)
  # define training arguments
  training_args = TrainingArguments(
      output_dir = "ov",
      max_steps = 100,
      per_device_train_batch_size = 2,
      gradient_accumulation_steps = 1,
      learning_rate = 2e-4,
      bf16 = True,
      logging_steps = 5,
      report_to = "wandb"
  )
  # initialize the trainer
  trainer = SFTTrainer(
      model="meta-llama/Llama-2-7b-chat-hf",
      train_dataset=dataset,
      dataset_text_field = "text",
      max_seq_length = 256,
      args = training_args
  )
  # start training
  trainer.train()
  # Make a 7B model memorise 50 examples with loss less than 50%, this is about plumbing
import random
import wandb
import torch
from datasets import load_datasets
from trl import SFTTrainer
from transformers import TrainingArguments

# initialize weights and biases for experiment tracking
wandb.init(projects="pb1-overfit", name=f"run-{random.randit(0,999)}")
# Load a small subset of the dataset (50 examples)
dataset= load_dataset("codefuse-ai/CodeExercise-Python-27k", split="train[:50]")

#Format the dataset to create instruction-response pairs
def format_example(example):
  return {
      "text": f"### Question: {example['chat_rounds'][0]['content']}\n### Answer: {example['chat_rounds'][1]['content']}"
  }
  dataset = dataset.map(format_example)
  # define training arguments
  training_args = TrainingArguments(
      output_dir = "ov",
      max_steps = 100,
      per_device_train_batch_size = 2,
      gradient_accumulation_steps = 1,
      learning_rate = 2e-4,
      bf16 = True,
      logging_steps = 5,
      report_to = "wandb"
  )
  # initialize the trainer
  trainer = SFTTrainer(
      model="meta-llama/Llama-2-7b-chat-hf",
      train_dataset=dataset,
      dataset_text_field = "text",
      max_seq_length = 256,
      args = training_args
  )
  # start training
  trainer.train()
  # Make a 7B model memorise 50 examples with loss less than 50%, this is about plumbing
import random
import wandb
import torch
from datasets import load_datasets
from trl import SFTTrainer
from transformers import TrainingArguments

# initialize weights and biases for experiment tracking
wandb.init(projects="pb1-overfit", name=f"run-{random.randit(0,999)}")
# Load a small subset of the dataset (50 examples)
dataset= load_dataset("codefuse-ai/CodeExercise-Python-27k", split="train[:50]")

#Format the dataset to create instruction-response pairs
def format_example(example):
  return {
      "text": f"### Question: {example['chat_rounds'][0]['content']}\n### Answer: {example['chat_rounds'][1]['content']}"
  }
  dataset = dataset.map(format_example)
  # define training arguments
  training_args = TrainingArguments(
      output_dir = "ov",
      max_steps = 100,
      per_device_train_batch_size = 2,
      gradient_accumulation_steps = 1,
      learning_rate = 2e-4,
      bf16 = True,
      logging_steps = 5,
      report_to = "wandb"
  )
  # initialize the trainer
  trainer = SFTTrainer(
      model="meta-llama/Llama-2-7b-chat-hf",
      train_dataset=dataset,
      dataset_text_field = "text",
      max_seq_length = 256,
      args = training_args
  )
  # start training
  trainer.train()
  # Make a 7B model memorise 50 examples with loss less than 50%, this is about plumbing
import random
import wandb
import torch
from datasets import load_datasets
from trl import SFTTrainer
from transformers import TrainingArguments

# initialize weights and biases for experiment tracking
wandb.init(projects="pb1-overfit", name=f"run-{random.randit(0,999)}")
# Load a small subset of the dataset (50 examples)
dataset= load_dataset("codefuse-ai/CodeExercise-Python-27k", split="train[:50]")

#Format the dataset to create instruction-response pairs
def format_example(example):
  return {
      "text": f"### Question: {example['chat_rounds'][0]['content']}\n### Answer: {example['chat_rounds'][1]['content']}"
  }
  dataset = dataset.map(format_example)
  # define training arguments
  training_args = TrainingArguments(
      output_dir = "ov",
      max_steps = 100,
      per_device_train_batch_size = 2,
      gradient_accumulation_steps = 1,
      learning_rate = 2e-4,
      bf16 = True,
      logging_steps = 5,
      report_to = "wandb"
  )
  # initialize the trainer
  trainer = SFTTrainer(
      model="meta-llama/Llama-2-7b-chat-hf",
      train_dataset=dataset,
      dataset_text_field = "text",
      max_seq_length = 256,
      args = training_args
  )
  # start training
  trainer.train()
  # Make a 7B model memorise 50 examples with loss less than 50%, this is about plumbing
import random
import wandb
import torch
from datasets import load_datasets
from trl import SFTTrainer
from transformers import TrainingArguments

# initialize weights and biases for experiment tracking
wandb.init(projects="pb1-overfit", name=f"run-{random.randit(0,999)}")
# Load a small subset of the dataset (50 examples)
dataset= load_dataset("codefuse-ai/CodeExercise-Python-27k", split="train[:50]")

#Format the dataset to create instruction-response pairs
def format_example(example):
  return {
      "text": f"### Question: {example['chat_rounds'][0]['content']}\n### Answer: {example['chat_rounds'][1]['content']}"
  }
  dataset = dataset.map(format_example)
  # define training arguments
  training_args = TrainingArguments(
      output_dir = "ov",
      max_steps = 100,
      per_device_train_batch_size = 2,
      gradient_accumulation_steps = 1,
      learning_rate = 2e-4,
      bf16 = True,
      logging_steps = 5,
      report_to = "wandb"
  )
  # initialize the trainer
  trainer = SFTTrainer(
      model="meta-llama/Llama-2-7b-chat-hf",
      train_dataset=dataset,
      dataset_text_field = "text",
      max_seq_length = 256,
      args = training_args
  )
  # start training
  trainer.train()
  # Make a 7B model memorise 50 examples with loss less than 50%, this is about plumbing
import random
import wandb
import torch
from datasets import load_datasets
from trl import SFTTrainer
from transformers import TrainingArguments

# initialize weights and biases for experiment tracking
wandb.init(projects="pb1-overfit", name=f"run-{random.randit(0,999)}")
# Load a small subset of the dataset (50 examples)
dataset= load_dataset("codefuse-ai/CodeExercise-Python-27k", split="train[:50]")

#Format the dataset to create instruction-response pairs
def format_example(example):
  return {
      "text": f"### Question: {example['chat_rounds'][0]['content']}\n### Answer: {example['chat_rounds'][1]['content']}"
  }
  dataset = dataset.map(format_example)
  # define training arguments
  training_args = TrainingArguments(
      output_dir = "ov",
      max_steps = 100,
      per_device_train_batch_size = 2,
      gradient_accumulation_steps = 1,
      learning_rate = 2e-4,
      bf16 = True,
      logging_steps = 5,
      report_to = "wandb"
  )
  # initialize the trainer
  trainer = SFTTrainer(
      model="meta-llama/Llama-2-7b-chat-hf",
      train_dataset=dataset,
      dataset_text_field = "text",
      max_seq_length = 256,
      args = training_args
  )
  # start training
  trainer.train()
  # Make a 7B model memorise 50 examples with loss less than 50%, this is about plumbing
import random
import wandb
import torch
from datasets import load_datasets
from trl import SFTTrainer
from transformers import TrainingArguments

# initialize weights and biases for experiment tracking
wandb.init(projects="pb1-overfit", name=f"run-{random.randit(0,999)}")
# Load a small subset of the dataset (50 examples)
dataset= load_dataset("codefuse-ai/CodeExercise-Python-27k", split="train[:50]")

#Format the dataset to create instruction-response pairs
def format_example(example):
  return {
      "text": f"### Question: {example['chat_rounds'][0]['content']}\n### Answer: {example['chat_rounds'][1]['content']}"
  }
  dataset = dataset.map(format_example)
  # define training arguments
  training_args = TrainingArguments(
      output_dir = "ov",
      max_steps = 100,
      per_device_train_batch_size = 2,
      gradient_accumulation_steps = 1,
      learning_rate = 2e-4,
      bf16 = True,
      logging_steps = 5,
      report_to = "wandb"
  )
  # initialize the trainer
  trainer = SFTTrainer(
      model="meta-llama/Llama-2-7b-chat-hf",
      train_dataset=dataset,
      dataset_text_field = "text",
      max_seq_length = 256,
      args = training_args
  )
  # start training
  trainer.train()
  !pip install -q bitsandbytes datasets accelerate loralib
!pip install -q git+https://github.com/huggingface/transformers.git@main git+https://github.com/huggingface/peft.git
#setup the model
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "bigscience/bloom-7b1",
    load_in_8bit=True,
    device_map='auto',
)
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-7b1")

# Freezing the original weights
for param in model.parameters():
    param.requires_grad = False # freeze the model -- train adapters later
    if param.ndim == 1:
      #cast the small parameters (eg. layernorm) to fp32 for stability
      param.data = param.data.to(torch.float32)
model.gradient_checkpointing_enable() # reduce number of stored activations
model.enable_input_require_grads()

class CastOutputToFLoat(nn.Sequential):
  def forward(self, x): return super().forward(x).to(torch.float32)
model.lm_head = CastOutputToFloat(model.lm_head)

#setting up the LoRa adapters
def print_trainable_parameters(model):
  # prints the number of trainable parameters in the model
  trainable_params = 0
  all_param = 0
  for _, param in model.named_parameters():
    all_param += param.numel()
    if param.requires_grad:
      trainable_params += param.numel()
  print(
      f"trainable param: {trainable_params} || all param: {all_params} || trainable%: {100 * trainable_params / all_params}"
  )
  from peft import LoraConfig, get_peft_model
  config = LoraConfig(
      r = 16, # attention heads
      lora_alpha = 32, # alpha scaling
      # target_modules = ["q_proj", "v_proj"], if you know the, what???
      lora_dropout=0.05,
      bias="None",
      task_type="CAUSAL_LM" # set this for CLM or SEQ2SEQ
  )
  model = get_peft_model(model, config)
  print_trainable_parameters(model)

  # data
  import transformers
  from datasets import load_dataset
  data=load_dataset("Abirate/english_quotes")
  def merge_columns(example):
    example["prediction"] = example["quote"] + " ->" + str(example["tags"])
    return example

  data['train'] = data['train'].map(merge_columns)
  data['train']["prediction"][:5]
  data['train'][0]
  data = data.map(lambda samples: tokenizer(samples['prediction']), batched=True)

  # Training
  trainer = transformer.Trainer(
      model = model,
      train_dataset=data['train'],
      args=transformers.TrainingArguments(
          per_device_train_batch_size = 4,
          gradient_accumulation_steps = 4,
          warmup_steps = 100,
          max_steps = 200,
          learning_rate = 2e-4,
          #bf16 = True,
          fp16 = True,
          logging_steps= 1,
          output_dir = 'outputs'
      ),
      data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
  )
  model.config.use_cache = False # silence the warnings, please re-enable for inference!
  trainer.train()

  # share adapters on the hub
  model.push_to_hub("samwit/bloom-7b1-lora-tagger",
                    use_auth_token=True,
                    commit_message="basic training",
                    private=True)
  # Load adapters from the hub
  import torch
  from peft import PeftModel, PeftConfig
  from transformers import AutoModelForCausalLM, AutoTokenizer
  peft_model_id = "samwit/bloom-7b1-lora-tagger"
  config = PeftConfig.from_pretrained(peft_model_id)
  model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)

  # Load the Lora Model
  model = PeftModel.from_pretrained(model, peft_model_id)
  # Inference
  bath = tokenizer(" Training models with PEFT and LoRa is cool ->: ", return_tensors='pt')
  with torch.cuda.amp.autocast():
    output_tokens = model.generate(**batch, max_new_tokens=50)
  print('\n\n', tokenizer.decode(output_tokens[0], skip_special_tokens=True))
  # setup the model
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # choose which GPU to use (0 = first GPU)
import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

# Load BLOOM-7B1 in 8-bit to save memory
model = AutoModelForCausalLM.from_pretrained(
    "bigscience/bloom-7b1",
    load_in_8bit=True, #quantized loading -> fits on consumer GPUs
    device_maps = "auto", # automatically places layers on available devices

)
# tokenizer for bloom
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-7b1")
# freeze the original model weights
for param in model.parameters():
  param.requires_grad = False # freeze base model weights (now lora will handle training)
  if param.ndim == 1:
    # keep small param like layernorm in fp32 for numerical stability
    param.data = param.data.to(torch.float32)
# enable memory savings during training
model.gradient_checkpointing_enable() # reduces stored activations at cost of extra compute
model.enable_input_require_grads() # allows grads to flow through inputs (needed for LoRa )

# custom wrapper to ensure outputs are in float32
class CastOutputToFloat(nn.Sequential):
  def forward(self, x):
    return super().forward(x).to(torch.float32)
# replace LM head with safe float32 head
model.lm_head = CastOutputToFloat32(model.lm_head)

# settting up lora adapters
from peft import LoraConfig, get_peft_model

def print_trainable_parameters(model):
  #utility to show how many params are actually trainable
  trainable_params = 0
  all_param = 0
  for _, param in model.named_parameters():
    all_params += param.numel()
    if param.requires_grad:
      trainable_params += param.numel()
  print(
      f"trainable param: {trainable_params} || all Params: {all_params}"
      f"trainable%: {100 * trainable_param / all_params: .4f}"
  )
# lora config tweak these hyperparameters for experiments
config = LoraConfig(
    r = 16,
    lora_alpha=32,
    lora_dropouts=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Add lora adapters to the model
model = get_peft_model(model, config)
print_trainable_parameters(model)

#data prep
import transformers
from datasets import load_dataset

# load Dataset of english quotes
data = load_dataset("Abirate/english_quotes")
# merge columns: join quote + tags into a training example
def merge_columns(example):
  example["prediction"] = example["quote"] + " -> " + str(example["tags"])
  return example
data["train"] = data["train"].map(merge_columns)

# quick sanity check
print(data["train"]["prediction"][":5"])
print(data["train"][0])

# tokenize dataset
data = data.map(lambda samples: tokenize(samples["prediction"]), batched= True)

# training loop
trainer = transformers.Trainer(
    model = model,
    train_dataset=data["train"],

)
from huggingface_hub import login
login(new_session=False)
# args.toml
model_name = "meta-llama/Llama-2-7b-chat-hf"
dataset_name = "codefuse-ai/CodeExercise-Python-27k"
lora_r = 16
lora_alpha = 32
learning_rate = 2e-4
per_device_train_batch_size = 4
max_steps = 200
quant_type = "nf4"
# Create and write to a file
file_content = """model_name = "meta-llama/Llama-2-7b-chat-hf"
dataset_name = "lvwerra/stack-exchange-paired"
lora_r = 8
lora_alpha = 16
learning_rate = 2e-4
per_device_train_batch_size = 1
max_steps = 200
quant_type = "nf4"
max_seq_length = 512"""

with open("args.toml", "w") as f:
    f.write(file_content)

print("File 'args.toml' created successfully.")
!pip install wandb
import wandb
wandb.login()
!pip install datasets transformers peft trl
!pip install -U bitsandbytes
# run.py
import tomllib, torch, wandb
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig as bnb, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

with open("args.toml", "rb") as f: cfg = tomllib.load(f)
wandb.init(project="law-of-100", config=cfg)

bnbv = bnb(
    load_in_4bit = True,
    bnb_4bit_quant_type = cfg["quant_type"],
    bnb_4bit_compute_dtype = torch.bfloat16,
    bnb_4bit_use_double_quant = True
)
model = AutoModelForCausalLM.from_pretrained(cfg["model_name"],
quantization_config = bnbv, device_map = "auto")
tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
tokenizer.pad_token = tokenizer.eos_token

peft = LoraConfig(r=cfg["lora_r"],
lora_alpha = cfg["lora_alpha"], target_modules = ["q_proj", "v_proj"], lora_dropout = 0.05, task_type = "CAUSAL_LM")
model = get_peft_model(model, peft)

raw = load_dataset(cfg["dataset_name"], streaming=True)
# print(raw.keys()) # Removed the diagnostic print
def fmt(batch): return {"text": f"### Human: {batch['chat_rounds'][0]['content']}\n### Assistant: {batch['chat_rounds'][1]['content']}"}

train = raw['default']['train'].shuffle().take(cfg["max_steps"]*cfg["per_device_train_batch_size"]).map(fmt)

args = TrainingArguments(output_dir = "out", max_steps = cfg["max_steps"], learning_rate = cfg["learning_rate"], per_device_train_batch_size = cfg["per_device_train_batch_size"], gradient_accumulation_steps = 2, fp16 = False, bf16 = True, logging_steps = 10, save_steps = 100, report_to = "wandb")
trainer = SFTTrainer(model = model, train_dataset = train, args = args, dataset_text_field = "text", max_seq_length = cfg["max_seq_length"])
trainer.train()
trainer.save_model("adapter")
"""Waiting for wandb.init()...
Tracking run with wandb version 0.21.3
Run data is saved locally in /content/wandb/run-20250913_150803-5yg2x7dl
Syncing run laced-thunder-19 to Weights & Biases (docs)
View project at https://wandb.ai/avedant34-freelancer/law-of-100
View run at https://wandb.ai/avedant34-freelancer/law-of-100/runs/5yg2x7dl
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipython-input-4099975426.py in <cell line: 0>()
     15     bnb_4bit_use_double_quant = True
     16 )
---> 17 model = AutoModelForCausalLM.from_pretrained(cfg["model_name"],
     18 quantization_config = bnbv, device_map = "auto")
     19 tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])

4 frames
/usr/local/lib/python3.12/dist-packages/transformers/quantizers/quantizer_bnb_4bit.py in validate_environment(self, *args, **kwargs)
    115                 pass
    116             elif "cpu" in device_map_without_lm_head.values() or "disk" in device_map_without_lm_head.values():
--> 117                 raise ValueError(
    118                     "Some modules are dispatched on the CPU or the disk. Make sure you have enough GPU RAM to fit the "
    119                     "quantized model. If you want to dispatch the model on the CPU or the disk while keeping these modules "

ValueError: Some modules are dispatched on the CPU or the disk. Make sure you have enough GPU RAM to fit the quantized model. If you want to dispatch the model on the CPU or the disk while keeping these modules in 32-bit, you need to set `llm_int8_enable_fp32_cpu_offload=True` and pass a custom `device_map` to `from_pretrained`. Check https://huggingface.co/docs/transformers/main/en/main_classes/quantization#offload-between-cpu-and-gpu for more details. """
from datasets import get_dataset_config_names
import tomllib

with open("args.toml", "rb") as f:
    cfg = tomllib.load(f)

dataset_name = cfg["dataset_name"]
configs = get_dataset_config_names(dataset_name)
print(f"Available configurations/splits for {dataset_name}: {configs}")
# eval.py
import pandas as pd, subprocess, tempfile, os, torch, evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtypes=torch.bfloat16, device_map = "auto")
model = PeftModel.from_pretrained(base, "adapter")
model = model.merge_and_unload()
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
rouge = evaluate.load("rouge")
test = load_dataset("codefuse-ai/CodeExercise-Python-27k", split = "test[:100]")
preds, refs = [], []
for row in test:
  prompt = f"### Human:{row['chat_rounds'][0]['content']}\n### Assistant:"
  inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
  out = model.generate(**input, max_new_tokens=128, do_sample=False)
  preds.append(tokenizer.decode(out[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True))
  refs.append(row['chat_rounds'][1]['content'])
r = rouge.compute(predictions=preds, references=refs, use_stemmer=True)

# unit test accuracy
correct = 0
for p, ref in zip(preds, refs):
  with tempfile.NameTemporaryFile(mode="w", suffix=".py", delete=False) as f:
    f.write(p+"\n"+ref)
    tmp =f.name
  correct += subprocess.run(["python",tmp], capture_output=True).return ncode==0
  os.unlink(tmp)
acc = correct/len(preds)
pd.DataFrame([{"lora_r":cfg["lora_r"], "lr":cfg["learning_rate"], "ROGUE-L":r["rougeL"],"code_acc":acc}]).to_csv("results.csv", mode="a",header=not os.path.exists("results.csv"))
#app.py
import gradio as gr, torch
from transformers import pipeline
pipe = pipeline("text-generation", model="merged-model", torch_dype=torch.bfloat16, device_map="auto")
def gen(prompt): return pipe(prompt, max_new_tokens=256, do_sample = True, temperature=0.7)[0]["generated_text"]
gr.Interface(fn=gen, inputs="text", outputs="text", title="Python-LoRA-100").launch()
!pip install -U trl datasets bitsandbytes
import os
import torch
from contextlib import nullcontext
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer
bnb_config = BitsAndBytesConfig(
    load_in_4bits=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float32
)
repo_id = 'microsoft/Phi-3-mini-4k-instruct'
model = AutoModelForCausalLM.from_pretrained(repo_id, device_map="cuda:0",quantization_config=bnb_config)
"""/usr/local/lib/python3.12/dist-packages/huggingface_hub/utils/_auth.py:94: UserWarning: 
The secret `HF_TOKEN` does not exist in your Colab secrets.
To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
You will be able to reuse this secret in all of your notebooks.
Please note that authentication is recommended but still optional to access public models or datasets.
  warnings.warn(
config.json: 100%
 967/967 [00:00<00:00, 48.2kB/s]
WARNING:bitsandbytes.cextension:The 8-bit optimizer is not available on your device, only available on CUDA for now.
model.safetensors.index.json: 
 16.5k/? [00:00<00:00, 1.50MB/s]
Fetching 2 files: 100%
 2/2 [02:36<00:00, 156.35s/it]
model-00002-of-00002.safetensors: 100%
 2.67G/2.67G [01:50<00:00, 16.9MB/s]
model-00001-of-00002.safetensors: 100%
 4.97G/4.97G [02:36<00:00, 49.0MB/s]
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipython-input-2842719920.py in <cell line: 0>()
      6 )
      7 repo_id = 'microsoft/Phi-3-mini-4k-instruct'
----> 8 model = AutoModelForCausalLM.from_pretrained(repo_id, device_map="cuda:0",quantization_config=bnb_config)

7 frames
/usr/local/lib/python3.12/dist-packages/torch/cuda/__init__.py in _lazy_init()
    410         if "CUDA_MODULE_LOADING" not in os.environ:
    411             os.environ["CUDA_MODULE_LOADING"] = "LAZY"
--> 412         torch._C._cuda_init()
    413         # Some of the queued calls may reentrantly call _lazy_init();
    414         # we need to just return without initializing in that case.

RuntimeError: Found no NVIDIA driver on your system. Please check that you have an NVIDIA GPU and installed a driver from http://www.nvidia.com/Download/index.aspx"""
print(model.get_memory_footprint()//1e6)
"""---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipython-input-425917604.py in <cell line: 0>()
----> 1 print(model.get_memory_footprint()//1e6)

NameError: name 'model' is not defined"""
model
"""Phi3ForCausalLM(
  (model): Phi3Model(
    (embed_tokens): Embedding(32064, 3072, padding_idx=32000)
    (layers): ModuleList(
      (0-31): 32 x Phi3DecoderLayer(
        (self_attn): Phi3Attention(
          (o_proj): Linear4bit(in_features=3072, out_features=3072, bias=False)
          (qkv_proj): Linear4bit(in_features=3072, out_features=9216, bias=False)
        )
        (mlp): Phi3MLP(
          (gate_up_proj): Linear4bit(in_features=3072, out_features=16384, bias=False)
          (down_proj): Linear4bit(in_features=8192, out_features=3072, bias=False)
          (activation_fn): SiLU()
        )
        (input_layernorm): Phi3RMSNorm((3072,), eps=1e-05)
        (post_attention_layernorm): Phi3RMSNorm((3072,), eps=1e-05)
        (resid_attn_dropout): Dropout(p=0.0, inplace=False)
        (resid_mlp_dropout): Dropout(p=0.0, inplace=False)
      )
    )
    (norm): Phi3RMSNorm((3072,), eps=1e-05)
    (rotary_emb): Phi3RotaryEmbedding()
  )
  (lm_head): Linear(in_features=3072, out_features=32064, bias=False)
)"""
model = prepare_model_for_kbit_training(model)
config = LoraConfig(
    r = 8,
    lora_alpha=16,
    bias="none",
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
    target_modules=['o_proj', 'qkv_proj', 'gate_up_proj', 'down_proj']
)
model = get_peft_model(model, config)
model
"""PeftModelForCausalLM(
  (base_model): LoraModel(
    (model): Phi3ForCausalLM(
      (model): Phi3Model(
        (embed_tokens): Embedding(32064, 3072, padding_idx=32000)
        (layers): ModuleList(
          (0-31): 32 x Phi3DecoderLayer(
            (self_attn): Phi3Attention(
              (o_proj): lora.Linear4bit(
                (base_layer): Linear4bit(in_features=3072, out_features=3072, bias=False)
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.05, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=3072, out_features=8, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=8, out_features=3072, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (lora_magnitude_vector): ModuleDict()
              )
              (qkv_proj): lora.Linear4bit(
                (base_layer): Linear4bit(in_features=3072, out_features=9216, bias=False)
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.05, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=3072, out_features=8, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=8, out_features=9216, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (lora_magnitude_vector): ModuleDict()
              )
            )
            (mlp): Phi3MLP(
              (gate_up_proj): lora.Linear4bit(
                (base_layer): Linear4bit(in_features=3072, out_features=16384, bias=False)
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.05, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=3072, out_features=8, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=8, out_features=16384, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (lora_magnitude_vector): ModuleDict()
              )
              (down_proj): lora.Linear4bit(
                (base_layer): Linear4bit(in_features=8192, out_features=3072, bias=False)
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.05, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=8192, out_features=8, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=8, out_features=3072, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (lora_magnitude_vector): ModuleDict()
              )
              (activation_fn): SiLU()
            )
            (input_layernorm): Phi3RMSNorm((3072,), eps=1e-05)
            (post_attention_layernorm): Phi3RMSNorm((3072,), eps=1e-05)
            (resid_attn_dropout): Dropout(p=0.0, inplace=False)
            (resid_mlp_dropout): Dropout(p=0.0, inplace=False)
          )
        )
        (norm): Phi3RMSNorm((3072,), eps=1e-05)
        (rotary_emb): Phi3RotaryEmbedding()
      )
      (lm_head): Linear(in_features=3072, out_features=32064, bias=False)
    )
  )
)"""
print(model.get_memory_footprint()//1e6)
trainable_parms, tot_parms = model.get_nb_trainable_parameters()
print(f'Trainable parameters:       {trainable_parms/1e6:.2f}M')
print(f'Trainable parameters:       {tot_parms/1e6:.2f}M')
print(f'Trainable parameters:       {100*trainable_parms//tot_parms:.2f}%')
""""""
import os
import torch
import json
import re
import os
import pandas as pd
from pprint import pprint
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from datasets import Dataset, load_dataset
from huggingface_hub import notebook_login
from peft import LoraConfig, PeftModel
from transformers import (
    AutoModelForCausalLM, # AutoModel for language modeling tasks
    AutoTokenizer, # AutoTokenizer for tokenization
    BitsAndBytesConfig, # Configuration for BitsAndBytes
    TrainingArguments # Training arguments for model training
)
from trl import SFTTrainer # SFTTrainer for supervised fine-tuning
# Data prepration
Import pandas as pd
import json
data = pd.read_csv('Airline-Sentiment-2-w-AA.csv', header=0, encodings='iso-8859-1')
data = data.sample(n=1000, random_state=1)

# Function to format each row into the desired JSON structure
def format_row(row):
  return {
      "text": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. ### Instruction: Based on the input sentence provided below, classify its sentiment with one of classes as positive, negative or neutral. ### Input: {} ### Response: {}".format(row['text'], row['airline_sentiment'])
  }
  formatted_data = data.apply(format_row, axis=1).tolist()
# save the formatted data to JSON file
with open('formatted_data.json', 'w', encodings='utf-8') as f:
    json.dump(formatted_data, f, indent=4)
print("data has been successfully formatted and saved to formatted_data.json")

#Alpaca Style
# Below I wrote two functions to format the data and load it into a format suitable for training.
# this function reads the data from the input file and writes the text data to the output file
def format_data(input_file, output_file):
  with open(input_file, 'r', encodings='utf-8') as file:
    data = json.load(file)
  formatted_data = [item['text'] for item in data] # extract the 'text' field from each item
  with open(output_file, 'w', encodings='utf-8') as file:
    file.write('\n'.json(formatted_data))
#This function loads the data from a text file and processed it into a format suitable for training
def load_and_process_data(file_path):
  with open(file_path, 'r', encodings='utf-8') as file: # read the text file
    lines = file.readlines()
  data = [{'text': line.strip()} for line in lines if line.strip()] # create a list of dictionaries
  df = pd.DataFrame(data)
  dataset = Dataset.from_pandas(df)
  return dataset
# Files we are going to use
input_json_file = 'formatted_data.json'
output_json_file = 'alpaca_output.json'

# Format the new data and write it to a file
format_data(input_json_file, output_txt_file)

# Load the formatted data and prepare it for training
file_path = output_txt_file
dataset = load_and_process_data(file_path)

#Split the dataset into train and test sets, and then split the test set into validation and test sets
# also shuffle for better performance
train_test_split = dataset.train_test_split(test_size=0.2, shuffle=True)
validation_test_split = train_test_split['test'].train_test_split(test_size=0.5, shuffle=True)
# Organize the final splits into a Dict for easy access during training
dataset = {
    "train": train_test_split['train'],
    "validation": validation_test_split['train'], # 50% of the combined test set
    "test": validation_test_split['test'] # 50% of the combined test set

}
# Model Selection
from huggingface_hub import interpreter_login
interpreter_login()

# this guy locally downloaded the model and tokenizer to avoid downloading it every time
def create_model_and_tokenizer(model_path, tokenizer_path):
  bnb_config = BitsAndBytesConfig( # A configuration object that specifies setting for loading the model in a quantized format
  load_in_4bits=True, # the model should be loaded in 4-bit precision, which reduces memory usage and can speed up computation
  bnb_4bit_quant_type="nf4", # Specifies the type of 4-bit quantization to use. "nf4" is a specific quantization type
  bnb_4bit_compute_dtype=torch.float16, # specifies the data type for computation

  )
  model = AutoModelForCausalLM.from_pretrained(
    model_path,
    use_safetensors=True, # safetensors is a safe and fast file format for storing and loading tensors
    quantization_config=bnb_config,
    trust_remote_code=True,
    device_map="auto", # Automatically assigns the model to the appropriate device
  # A function from the hugging face transformers library that loads a pre-trained causal language model

  )

  # A tokenizer is in charge of praparing the inputs for a model
  tokenizer = AutoTokenizer.from_pretrained(tokenizer_path) # Transformers library that loads a tokenizer
  tokenizer.pad_token = tokenizer.eos_token # sets the padding token to be the same as the end-of-sequence token
  tokenizer.padding_side = "right" # specifies that padding should be added to the right side of the sequence
  # why are we padding??
  # dealing with sequence of varying lengths
  # the pad token is used to fill shorter sequences in a batch to match the longest sequence's length, ensuring uniform input size for model processing
  # eos_token: Uses one token to serve two purposes: indicate the end of meaningful content and fill space in shorter sequences to match the length of the longest sequence in a batch.
  # eos: the model only needs to learn to handle one special token (`<eos>`)
  # that can indicate the end of meaningful content. This reduces the number of rules the model needs to learn
  # takeaway: the distinction between padding and end of sequence needs to be minimal to simplify the model's decision-making process on when to stop generating text.
  return model, tokenizer
# Replace '_model_' and '_token_' with paths where you stored the model and tokenizer, if you cease to download it locally
model_path = '/finetune-llama/llama3-instruct-8b'
tokenizer_path = 'finetune-llama/llama3-instruct-8b'

model, tokenizer = create_model_and_tokenizer(model_path, tokenizer_path)
model. config.quantization_config.to_dict()
# Fine Tuning
lora_r = 64
# LoRA attention dimension: determines the dimension of the low-rank matrices used in
# attention layers. rank corresponds to the number of parameters in the adaptation
# layers -- the more parameters, the better it remembers, and the more complex
# things it can pick up. A higher rank allows more flexibitlity in the adaptation but increases the computational cost
# if too high: the model may overfit to the training data and perform poorly on new data.
# if too low: the model may not be able to capture the necessary patterns in the data and may underfit.
lora_alpha = 16 # Alpha parameter for lora scaling Alpha is a scaling factor (applied to the lora weights) it changes how the adaptation layer's weights affect the base model
# if too high: increased influence of the lora weights, potentially overpowering the base model
# if too low: the lora weights may not have a significant impact on the model's performance
lora_dropout = 0.1 # Drop out probablity for lora layers, dropout is a regularization technique that helps prevent overfitting by randomly setting a fraction of the output units to zero during training. if too high: then excessive regularization; many units are dropped, leading to a significant reduction in model capacity during trainig.
# if too low: few units are dropped, which may lead to overfitting
# its crucial to know where teh lora adaptations will be applied within the model
# specifies the modules in the model where the lora adaptations should be applied
# these are the projection matrices in the attention mechanism of a transformer
# model:
# q_proj: Query projection matrix
# k_proj: Key projection matrix
# v_proj: Value projection matrix
# o_proj: Output projection matrix
lora_target_modules = [
    "q_proj", # purpose: converts input embeddings into queries. Improves how
    # the model identifies and attends to important parts of the input.
    "k_proj", # Purpose: Converts input embeddings into keys. Enhances the
    # model's ability to match and highlight task-relevant patterns.
    "v_proj", # Purpose: Converts input embeddings into values for attention.
    # Refines the content, the model focuse on for the task # the reason that values are processed/called more quickly than rest of it at hand. so we call this idea attention. parses it as values for model to focus on.
    "o_proj", # Purpose: Transforms the attention-weighted sum back into the
    # input dimension. Fine-tunes the final output, making it more relevant to the task

]
peft_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    target_modules=lora_target_modules,
    bias="none",
    task_type="CAUSAL_LM"
)
pip install tensorboard # this is the board to catch checkpoints and logs during training
%load_ext tensorboard
%tensorboard --logdir llama3_sentiment_v2
OUTPUT_DIR = "llama3_sentiment_v2" # This is where the model checkpoints' and logs' during training
training_arguments = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size = 4, # batch size per GPU for training.
    # if too high, the model may run out of memory. if too low,
    #slow trainig per epoch
    gradient_accumulation_steps=2, # number of update steps to accumulate the gradients for.
    # if too high, then longer time between updates,
    # if too low, slower convergence
    optim="paged_adamw_32bits", # bnb.optim.adam8bit
    save_steps=25, # save checkpoint every X updates steps
    logging_steps=25, # log every X updates steps
    learning_rate=3e-4, # initial learning rate (AdamW optimizer)
    #fp16=True
    bf16=True,
    max_grad_norm=0.3,
    num_train_epochs=2,
    evaluation_strategy="steps",
    eval_steps=0.2,
    warmup_ratio=0.05, # used to gradually increase the learning rate at the
    # beginning of training. Allows the model to start with smaller,
    # safer updates, reducing the risk of instability in the early stages of training.
    save_strategy="epoch",
    group_by_length=True,
    report_to="tensorboard",
    save_safetensors=True,
    lr_scheduler_type="cosine", # Learning rate schedule
    seed=42,
)
# choose a fine-tuning technique: Supervised Fine-tuning (SFT) or Reinforcement Learning with human Feedback
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=4096, # it was 4096 and Halil changed it to 512 in (what)??
    tokenizer=tokenizer,
    args=training_arguments,
)
trainer.train()
# Fine-tuned model name
new_model = "llama3_sentiment_v2"
# save the fine-tuned model
trainer.model.save_pretrained(new_model)

# save train and test sets into path as csv
path = "llama3_sentiment_v2"
# Assuming `dataset["train"]` is a Hugging Face Dataset object
train_df = dataset["train"].to_pandas()
train.to_csv(f"{path}/train.csv", index=False)

# Repeat for validation and test datasets
validation_df = dataset["validation"].to_pandas()
validation_df.to_csv(f"{path}/validation.csv", index=False)
# llama for causal LM
# main model class
# Llamafor causal lm: this is the main model calss used for causal language modelling.
# it is built upon the llama architecture, which is optimized for autoregressive
# tasks like text generation.
#Still InCOMPLETE...
