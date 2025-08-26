# Scalable Multi-GPU Data Loader Strategy for Amazon Review Embedding Ingestion

## Objective

Efficiently load and embed millions of Amazon user reviews (JSONL) using all available GPUs, maximizing throughput, avoiding CUDA out-of-memory, and ensuring safe, lock-free concurrent insertion into PostgreSQL.

## High-Level Approach

1. **GPU Detection and Sharding**
   - Use `torch.cuda.device_count()` to get available GPU count: `N`.
   - Split input JSONL file into `N` equal parts (e.g., by number of lines).
     - For fast splits (line counts known), do this statically (e.g., with `split -l` or programmatically).
     - Each shard assigned to a single GPU index.

2. **Single Process per GPU, Pinned with CUDA_VISIBLE_DEVICES**
   - Use Python `multiprocessing` (or `subprocess`) to spawn `N` Python worker processes.
   - Each child process:
      - Sets `CUDA_VISIBLE_DEVICES` to its assigned GPU.
      - Loads its shard of the input file.
      - Loads SentenceTransformer (model loaded once per GPU, not per batch).
      - Streams data in moderate batches (e.g., 256–1024 reviews).
      - Embeds in batches (catching/capping to avoid OOM), yields NumPy/PyTorch tensors and corresponding row data.

3. **DB Insertion in Safe Batches**
   - Each worker:
      - Connects individually to Postgres.
      - Begins transaction for each batch.
      - Inserts/updates the batch.
      - Commits, then proceeds.
      - Always closes each transaction before starting next.
      - Optionally, use advisory locks or primary-key ranges to avoid overlap if needed (not usually required if records are disjoint).

4. **PyTorch Memory Hygiene**
   - After every batch:
      - Call `torch.cuda.empty_cache()`.
      - Delete reference to embedding tensor to free GPU memory before next batch.
      - If possible, use `with torch.no_grad():` everywhere for embedding.

5. **Coordinator/Recovery**
   - Main process can:
      - Monitor children for completion.
      - Collate/report progress.
      - Retry failed batches (optionally).
      - Collate global error log.

6. **Optional: Parallel File Split**
   - For huge files, do OS-level split first (avoid one master reading/partitioning all in RAM).
   - Use: `split -n l/N file.jsonl part_` or use Python to assign every Nth line to each worker’s input.

7. **NUMA and Mixed (CPU/GPU) Scaling**
   - For mixed clusters, extend above to allow CPU fallback, pin workers to CPU nodes, and adjust batch size downward.

## Usage Summary (Pseudocode)

1. Launch script:
    python3 multi_gpu_loader.py --reviews Cell_Phones_and_Accessories.jsonl --metadata meta_Cell_Phones_and_Accessories.jsonl

2. Script workflow:
    - Detect N GPUs
    - Split reviews file into N files: reviews_chunk_0.jsonl, ..., reviews_chunk_{N-1}.jsonl
    - Spawn N worker processes, each with:
        CUDA_VISIBLE_DEVICES=[i] python3 gpu_worker.py --reviews reviews_chunk_i.jsonl ...
    - Each worker loads only its chunk, batches, embeds, batch-inserts, repeats until complete.

## Best Practices to Avoid OOM and DB Locks

- Tune batch size experimentally per GPU (start with small batches).
- Do not let child process queue up all DB inserts into Postgres in one open transaction—commit after each batch.
- Set DB `statement_timeout` if needed for very slow batches.
- For extra safety, index DB columns on arrival to help query speed for FTS/embedding checks.
- Clear GPU cache after every batch.
- Use try/except/finally on all batch operations for error robustness.

## Final Notes

- This approach scales linearly with the number of GPUs and does not exceed VRAM on any device.
- All DB operations are parallel but independent; Postgres connection pool size can be increased if needed.
- To maximize DB speed, keep single-batch sizes moderate—let the DB accept many simultaneous small transactions, not a few gigantic ones.
