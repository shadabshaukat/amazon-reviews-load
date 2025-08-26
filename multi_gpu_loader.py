import os
import sys
import argparse
import torch
import subprocess
import math
import logging

def detect_gpus():
    return torch.cuda.device_count()

def split_jsonl(input_path, num_chunks, output_prefix='reviews_chunk_'):
    # Count total lines (not holding all in RAM)
    with open(input_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    lines_per_chunk = math.ceil(total_lines / num_chunks)

    out_files = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for i in range(num_chunks):
            chunk_path = f"{output_prefix}{i}.jsonl"
            out_files.append(chunk_path)
            with open(chunk_path, 'w', encoding='utf-8') as cf:
                for _ in range(lines_per_chunk):
                    line = f.readline()
                    if not line:
                        break
                    cf.write(line)
    return out_files

def launch_worker(chunk_path, gpu_idx, meta_path, db_env_path, batch_size=128, skip_missing_metadata=False):
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_idx)
    env['WORKER_ID'] = str(gpu_idx)
    if db_env_path:
        from dotenv import dotenv_values
        config = dotenv_values(db_env_path)
        env.update(config)
    cmd = [
        sys.executable, 'gpu_embed_worker.py',
        '--reviews', chunk_path,
        '--metadata', meta_path,
        '--batch-size', str(batch_size)
    ]
    if skip_missing_metadata:
        cmd.append('--skip-missing-metadata')
    return subprocess.Popen(cmd, env=env)

def main():
    parser = argparse.ArgumentParser(description="Multi-GPU Amazon Reviews Loader Orchestrator")
    parser.add_argument('--reviews', required=True, help="Input reviews JSONL file")
    parser.add_argument('--metadata', required=True, help="Input metadata JSONL file")
    parser.add_argument('--db-dotenv', default=".env", help="Path to .env with DB creds to forward to workers")
    parser.add_argument('--batch-size', type=int, default=128, help="Batch size per GPU")
    parser.add_argument('--skip-missing-metadata', action='store_true', help="Skip reviews with no parent FK")
    parser.add_argument('--max-gpus', type=int, default=None, help="(Optional) Cap number of GPUs used")
    args = parser.parse_args()

    logging.basicConfig(format='[%(asctime)s][%(levelname)s] %(message)s', level=logging.INFO)
    num_gpus = detect_gpus()
    if not num_gpus:
        logging.error("No GPUs detected.")
        sys.exit(1)
    if args.max_gpus:
        num_gpus = min(num_gpus, args.max_gpus)

    logging.info(f"Detected {num_gpus} GPUs.")
    chunk_files = split_jsonl(args.reviews, num_gpus)
    logging.info(f"Split '{args.reviews}' into {len(chunk_files)} chunks.")

    workers = []
    for gpu_idx, chunk_file in enumerate(chunk_files):
        logging.info(f"Launching worker on GPU {gpu_idx}: {chunk_file}")
        proc = launch_worker(chunk_file, gpu_idx, args.metadata, args.db_dotenv, batch_size=args.batch_size, skip_missing_metadata=args.skip_missing_metadata)
        workers.append(proc)

    for proc in workers:
        proc.wait()

    logging.info("Multi-GPU review loading complete.")

if __name__ == "__main__":
    main()
