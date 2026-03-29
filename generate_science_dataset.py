import asyncio
import json
import random
import re
import logging
from pathlib import Path
from collections import deque
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import aiofiles

# ========================== HYPER-DETAILED CONFIGURATION ==========================
@dataclass
class Config:
    model_name: str = "HuggingFaceTB/SmolLM3-3B"  # Strong small model for math/reasoning (2026)
    load_in_4bit: bool = True
    max_new_tokens: int = 1800
    temperature: float = 0.55
    top_p: float = 0.92
    batch_samples: int = 800               # Safe per-run batch for 6h timeout on ubuntu-latest
    target_total: int = 100000
    max_concurrent: int = 1                # CPU-only safety
    retries: int = 5
    backoff_factor: float = 1.6
    data_dir: Path = Path("dataset_output")
    output_file: Optional[Path] = None
    progress_file: Path = Path("dataset_output/current_count.txt")

    def __post_init__(self):
        self.data_dir.mkdir(exist_ok=True, parents=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_file = self.data_dir / f"science_heavy_cpu_{timestamp}.jsonl"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)-8s | %(message)s',
            handlers=[
                logging.FileHandler(self.data_dir / f"run_log_{timestamp}.log", encoding="utf-8"),
                logging.StreamHandler()   # Real-time streaming in GitHub Actions logs
            ]
        )
        logging.info(f"Starting batch generation. Model: {self.model_name} | Batch size: {self.batch_samples} | Overall target: {self.target_total}")

config = Config()

# ========================== STEM TOPICS & EXPANSION ==========================
STEM_TOPICS_STARTERS: List[str] = [
    "derive escape velocity from first principles",
    "prove sqrt(2) is irrational using multiple methods",
    "explain enzyme catalysis via transition state theory with molecular details",
    "derive ideal gas law from kinetic molecular theory step-by-step",
    "derive Nernst equation from Gibbs free energy and reaction quotient",
    "solve and interpret damped harmonic oscillator differential equation",
    "explain semi-conservative DNA replication with all major enzymes",
    "derive time dilation in special relativity from light clock thought experiment",
    "prove fundamental theorem of calculus rigorously",
    "derive Michaelis-Menten kinetics from enzyme-substrate binding assumptions",
    "derive Schrödinger equation using variational principle",
    "prove Noether's theorem with physical implications",
    "explain Krebs cycle with detailed energetics and regulation",
    "derive Planck's law for blackbody radiation",
    "prove Bayes theorem from axioms with multiple derivations",
]

def expand_topic_seed(seed: str) -> List[str]:
    prefixes = [
        "derive", "prove", "explain the detailed mechanism of",
        "derive rigorously from first principles", "show using multiple approaches",
        "derive thermodynamically", "solve the differential equation for",
        "prove using contradiction and direct method", "explain quantum mechanically"
    ]
    variants = [f"{p} {seed}" for p in prefixes]
    depth_variants = [
        f"{seed} including limiting cases, dimensional analysis, and physical intuition",
        f"{seed} with historical context, sources, and modern applications",
        f"advanced graduate-level derivation and extensions of {seed}",
        f"{seed} with error analysis, common misconceptions, and micro-verifications",
    ]
    expanded = list(set(variants + depth_variants))
    random.shuffle(expanded)
    return expanded[:10]

SYSTEM_PROMPT = """You are an expert scientific reasoner. Output ONLY in this EXACT strict format — nothing before or after. Use heavy LaTeX for equations. Be hyper-detailed, multi-angle, rigorous. Zero hallucinations.

### [DECOMPOSITION]
Numbered hierarchical breakdown into atomic subproblems.

### [AXIOMS & PROVENANCE]
Bullet list of laws, constants, sources (textbooks/papers/years).

### [NESTED REASONING]
**Step 1: Title**
Detailed paragraph(s) + equations.

**Step 2: ...**
Deeper detail, alternatives, cross-checks, edge cases.

Continue deeply...

### [MICRO-VERIFICATION]
- Dimensional/units check ✓
- Limiting case ✓
- Known value match ✓
- Logical consistency ✓
- Alternative derivation ✓

### [FINAL ANSWER]
Concise \\boxed{result} + 1-sentence summary.
"""

# ========================== MODEL LOADING ==========================
def load_model():
    logging.info("Loading SmolLM3-3B-Instruct (4-bit quantized for CPU).")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    ) if config.load_in_4bit else None

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=quant_config,
        device_map="cpu",          # Force CPU for hosted runner
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    logging.info("Model loaded successfully (100% local, no cloud).")
    return model, tokenizer

# ========================== GENERATION ==========================
async def generate_one(model, tokenizer, instruction: str) -> Optional[Dict[str, Any]]:
    user_msg = f"Instruction: {instruction}\n\nFollow the format exactly. Make it longer, more detailed, and more rigorous than typical textbook solutions. Include edge cases and verifications."
    full_prompt = f"{SYSTEM_PROMPT}\n\n{user_msg}"

    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    for attempt in range(config.retries):
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            content = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

            if not content.startswith("### [DECOMPOSITION]") or "### [FINAL ANSWER]" not in content or "\\boxed" not in content:
                raise ValueError("Strict format violation")

            result = {"instruction": instruction, "input": "", "output": content}
            logging.info(f"SUCCESS: {instruction[:75]}...")
            return result
        except Exception as e:
            if attempt == config.retries - 1:
                logging.warning(f"Failed after {config.retries} attempts: {instruction[:100]}")
                return None
            await asyncio.sleep(config.backoff_factor ** attempt * 1.2)

    return None

# ========================== WORKER & TOPIC GENERATOR ==========================
async def worker(model, tokenizer, queue: asyncio.Queue, counter: list, lock: asyncio.Lock, total_generated: int):
    while True:
        try:
            instr = await queue.get()
        except asyncio.Queue.Empty:
            break

        result = await generate_one(model, tokenizer, instr)

        async with lock:
            counter[0] += 1
            current_total = total_generated + counter[0]
            if result and config.output_file:
                async with aiofiles.open(config.output_file, "a", encoding="utf-8") as f:
                    await f.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(f"[{current_total:6d}/{config.target_total}] Generated → {instr[:68]}...")  # Real-time stream

        queue.task_done()

async def topic_generator(seed_queue: deque, instr_queue: asyncio.Queue, max_to_generate: int):
    seen = set()
    generated_count = 0
    while generated_count < max_to_generate and len(seen) < max_to_generate * 1.5:
        if not seed_queue:
            break
        current = seed_queue.popleft()
        if current in seen:
            continue
        seen.add(current)
        for v in expand_topic_seed(current):
            if v not in seen and generated_count < max_to_generate:
                await instr_queue.put(v)
                seen.add(v)
                seed_queue.append(v)
                generated_count += 1
        if random.random() < 0.13:
            seed_queue.extend(random.sample(STEM_TOPICS_STARTERS, k=2))

# ========================== MAIN ==========================
async def main():
    logging.info("=== Science-Heavy Dataset Generation Started (GitHub-hosted CPU) ===")
    
    # Load current progress
    total_generated = 0
    if config.progress_file.exists():
        try:
            total_generated = int(config.progress_file.read_text().strip())
        except:
            total_generated = 0
    logging.info(f"Previous total: {total_generated} | Generating up to {config.batch_samples} more this run.")

    model, tokenizer = load_model()

    topic_q = deque(STEM_TOPICS_STARTERS)
    instr_q: asyncio.Queue = asyncio.Queue(maxsize=10)

    producer = asyncio.create_task(topic_generator(topic_q, instr_q, config.batch_samples))

    batch_counter = [0]
    lock = asyncio.Lock()
    workers = [asyncio.create_task(worker(model, tokenizer, instr_q, batch_counter, lock, total_generated))
               for _ in range(config.max_concurrent)]

    await producer
    await instr_q.join()

    for w in workers:
        w.cancel()

    new_total = total_generated + batch_counter[0]
    config.progress_file.write_text(str(new_total))

    logging.info(f"=== Batch finished. New total: {new_total}/{config.target_total} ===")
    logging.info(f"Dataset appended to: {config.output_file}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Stopped. Partial batch saved.")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
