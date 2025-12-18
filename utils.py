# utils.py
import os
import gc
import random
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline


def setup_reproducibility(seed=42):
    print(f"[SYSTEM] Setting random seed to {seed}...")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(model_name: str, use_4bit: bool | None = None):
    print(f"\n[SYSTEM] Loading model: {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    use_cuda = torch.cuda.is_available()
    device_map = "auto" if use_cuda else None
    torch_dtype = torch.float16 if use_cuda else torch.float32

    # ✅ Decide 4-bit without requiring config.py
    if use_4bit is None:
        # env var fallback: export USE_4BIT_QUANTIZATION=1
        use_4bit = os.getenv("USE_4BIT_QUANTIZATION", "0") in ("1", "true", "True")

    if use_4bit and use_cuda:
        print("   -> Mode: 4-bit Quantization (GPU)")
        try:
            from transformers import BitsAndBytesConfig
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device_map,
                quantization_config=bnb_cfg,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device_map,
                load_in_4bit=True,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
    else:
        print("   -> Mode: Full Precision")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )

    return model, tokenizer


def create_pipeline(model, tokenizer, temperature, max_tokens=512):
    gen_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=0.95,
        repetition_penalty=1.15,
        return_full_text=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return HuggingFacePipeline(pipeline=gen_pipe)


def clean_memory():
    print("[SYSTEM] Cleaning memory...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
