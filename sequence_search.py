"""
Distributed Protein Sequence Search

Searches query protein sequences against a FASTA database using k-mer
similarity — a distributed alternative to BLAST for fast approximate matching.

The database is split into chunks and searched in parallel across all available
CPUs. Each worker reads its chunk, parses sequences, and scores them against
the queries using shared k-mer content (Jaccard similarity).

Usage:
  # Search your sequences against UniRef90
  python sequence_search.py --query my_proteins.fasta

  # Test mode — picks sequences from the DB as queries
  python sequence_search.py --test

  # Custom database and parameters
  python sequence_search.py --query query.fasta --db /path/to/db.fasta \
      --kmer-size=4 --top-n=20
"""

import argparse
import os
import time

import numpy as np
import ray


# ---------------------------------------------------------------------------
# FASTA parsing
# ---------------------------------------------------------------------------

def parse_fasta(text):
    """Parse FASTA text into (header, sequence) pairs."""
    sequences = []
    header = None
    parts = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                sequences.append((header, "".join(parts)))
            header = line[1:].strip()
            parts = []
        else:
            parts.append(line)
    if header is not None:
        sequences.append((header, "".join(parts)))
    return sequences


def parse_fasta_from_bytes(data, start_offset):
    """Parse FASTA from a byte chunk, handling boundary alignment."""
    text = data.decode("utf-8", errors="replace")

    # If we're not at the start of the file, skip to the first '>' to avoid
    # partial sequences
    if start_offset > 0:
        idx = text.find("\n>")
        if idx == -1:
            return []
        text = text[idx + 1:]

    return parse_fasta(text)


# ---------------------------------------------------------------------------
# K-mer indexing
# ---------------------------------------------------------------------------

# Standard amino acid alphabet (20 + X for unknown)
AA_CHARS = "ACDEFGHIKLMNPQRSTVWXY"
AA_MAP = {c: i for i, c in enumerate(AA_CHARS)}
ALPHA_SIZE = len(AA_CHARS)


def seq_to_kmers(seq, k):
    """Extract the set of k-mer hashes from a protein sequence."""
    kmers = set()
    seq = seq.upper()
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i + k]
        # Skip k-mers with non-standard amino acids
        h = 0
        valid = True
        for c in kmer:
            idx = AA_MAP.get(c)
            if idx is None:
                valid = False
                break
            h = h * ALPHA_SIZE + idx
        if valid:
            kmers.add(h)
    return kmers


def jaccard(set_a, set_b):
    """Jaccard similarity between two sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Distributed search
# ---------------------------------------------------------------------------

@ray.remote
def search_chunk(db_path, start, end, query_kmers_list, query_headers, k, top_n):
    """Search a byte range of the database for matches to query sequences.

    Returns a list of (query_idx, score, db_header, db_length) tuples — the
    top_n best hits per query found in this chunk.
    """
    with open(db_path, "rb") as f:
        f.seek(start)
        data = f.read(end - start)

    db_seqs = parse_fasta_from_bytes(data, start)

    # Deserialize query k-mer sets
    query_kmer_sets = [set(ks) for ks in query_kmers_list]
    n_queries = len(query_kmer_sets)

    # Track top_n hits per query for this chunk
    # Each entry: (score, header, length)
    top_hits = [[] for _ in range(n_queries)]

    for db_header, db_seq in db_seqs:
        if len(db_seq) < k:
            continue
        db_kmers = seq_to_kmers(db_seq, k)

        for qi in range(n_queries):
            score = jaccard(query_kmer_sets[qi], db_kmers)
            if score > 0:
                hits = top_hits[qi]
                if len(hits) < top_n:
                    hits.append((score, db_header, len(db_seq)))
                elif score > hits[-1][0]:
                    hits[-1] = (score, db_header, len(db_seq))
                    hits.sort(key=lambda x: -x[0])

    # Flatten into return format
    results = []
    for qi in range(n_queries):
        for score, header, length in top_hits[qi]:
            results.append((qi, score, header, length))

    return results


def find_chunk_boundaries(db_path, num_chunks):
    """Split a FASTA file into roughly equal byte-range chunks aligned to
    sequence boundaries ('>').
    """
    file_size = os.path.getsize(db_path)
    chunk_size = file_size // num_chunks
    boundaries = [0]

    with open(db_path, "rb") as f:
        for i in range(1, num_chunks):
            target = i * chunk_size
            f.seek(target)
            # Read ahead to find next '>' at start of line
            buf = f.read(min(1_000_000, file_size - target))
            idx = buf.find(b"\n>")
            if idx == -1:
                continue
            boundaries.append(target + idx + 1)
    boundaries.append(file_size)
    return boundaries


# ---------------------------------------------------------------------------
# Test mode: extract sample queries from the database
# ---------------------------------------------------------------------------

def extract_test_queries(db_path, n=5, seed=42):
    """Read the first 10K sequences and pick n at random as test queries."""
    seqs = []
    with open(db_path, "r") as f:
        header = None
        parts = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header is not None:
                    seqs.append((header, "".join(parts)))
                    if len(seqs) >= 10000:
                        break
                header = line[1:].strip()
                parts = []
            else:
                parts.append(line)
        if header is not None and len(seqs) < 10000:
            seqs.append((header, "".join(parts)))

    rng = np.random.default_rng(seed)
    # Pick sequences of moderate length (100-500 aa)
    candidates = [(h, s) for h, s in seqs if 100 <= len(s) <= 500]
    if len(candidates) < n:
        candidates = seqs
    indices = rng.choice(len(candidates), size=min(n, len(candidates)), replace=False)
    return [candidates[i] for i in indices]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Distributed protein sequence search using k-mer similarity"
    )
    parser.add_argument("--query", type=str, default=None,
                        help="Query FASTA file with protein sequences to search")
    parser.add_argument("--test", action="store_true",
                        help="Test mode — pick 5 sequences from DB as queries")
    parser.add_argument("--db", type=str,
                        default="/misc/blast/db/uniref90/uniref90.fasta",
                        help="Database FASTA file (default: UniRef90)")
    parser.add_argument("--kmer-size", type=int, default=3,
                        help="K-mer size for similarity scoring (default: 3)")
    parser.add_argument("--top-n", type=int, default=10,
                        help="Report top N hits per query (default: 10)")
    parser.add_argument("--num-queries", type=int, default=5,
                        help="Number of test queries in --test mode (default: 5)")
    parser.add_argument("--chunk-mb", type=int, default=100,
                        help="Database chunk size in MB (default: 100)")
    args = parser.parse_args()

    if args.query is None and not args.test:
        parser.error("Either --query or --test is required")

    if not os.path.exists(args.db):
        print(f"ERROR: Database not found: {args.db}")
        return 1

    ray.init()
    resources = ray.cluster_resources()
    num_cpus = int(resources.get("CPU", 1))
    num_nodes = sum(1 for k in resources if k.startswith("node:") and "internal" not in k)

    db_size_gb = os.path.getsize(args.db) / (1024**3)

    print("=" * 70)
    print("DISTRIBUTED PROTEIN SEQUENCE SEARCH")
    print("=" * 70)
    print(f"Cluster:    {num_cpus} CPUs across {num_nodes} node(s)")
    print(f"Database:   {args.db} ({db_size_gb:.1f} GB)")
    print(f"K-mer size: {args.kmer_size}")
    print(f"Top hits:   {args.top_n} per query")
    print()

    # Load or generate query sequences
    if args.test:
        print(f"Test mode: extracting {args.num_queries} queries from database...")
        queries = extract_test_queries(args.db, n=args.num_queries)
    else:
        with open(args.query, "r") as f:
            queries = parse_fasta(f.read())
        if not queries:
            print(f"ERROR: No sequences found in {args.query}")
            return 1

    print(f"Queries:    {len(queries)} sequences")
    for header, seq in queries:
        name = header.split()[0] if header else "?"
        print(f"  {name} ({len(seq)} aa)")
    print()

    # Build k-mer sets for queries
    t_start = time.time()
    query_headers = [h for h, _ in queries]
    query_kmer_sets = [seq_to_kmers(seq, args.kmer_size) for _, seq in queries]
    # Serialize k-mer sets as lists for Ray
    query_kmers_serialized = [list(ks) for ks in query_kmer_sets]

    # Split database into chunks
    file_size = os.path.getsize(args.db)
    num_chunks = max(num_cpus, file_size // (args.chunk_mb * 1024 * 1024))
    num_chunks = min(num_chunks, 1000)  # cap at 1000 chunks

    print(f"Splitting database into {num_chunks} chunks...")
    boundaries = find_chunk_boundaries(args.db, num_chunks)
    actual_chunks = len(boundaries) - 1
    t_prep = time.time()
    print(f"Prepared {actual_chunks} chunks ({t_prep - t_start:.1f}s)")
    print()

    # Launch parallel search
    print(f"Searching {db_size_gb:.1f} GB across {num_cpus} CPUs...")
    db_path_ref = ray.put(args.db)
    query_kmers_ref = ray.put(query_kmers_serialized)
    query_headers_ref = ray.put(query_headers)

    futures = []
    for i in range(actual_chunks):
        f = search_chunk.remote(
            db_path_ref, boundaries[i], boundaries[i + 1],
            query_kmers_ref, query_headers_ref,
            args.kmer_size, args.top_n,
        )
        futures.append(f)

    # Collect results with progress
    all_results = [[] for _ in range(len(queries))]
    completed = 0
    total = len(futures)
    last_pct = -1

    while futures:
        done, futures = ray.wait(futures, num_returns=min(10, len(futures)))
        for ref in done:
            chunk_results = ray.get(ref)
            for qi, score, header, length in chunk_results:
                all_results[qi].append((score, header, length))
            completed += 1

        pct = 100 * completed // total
        if pct >= last_pct + 10:
            elapsed = time.time() - t_prep
            gb_done = db_size_gb * completed / total
            print(f"  {pct:3d}%  ({completed}/{total} chunks, "
                  f"{gb_done:.1f}/{db_size_gb:.1f} GB, {elapsed:.0f}s)")
            last_pct = pct

    t_search = time.time()

    # Merge and sort results per query
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()

    for qi, (header, seq) in enumerate(queries):
        name = header.split()[0] if header else f"query_{qi}"
        hits = all_results[qi]
        hits.sort(key=lambda x: -x[0])
        hits = hits[:args.top_n]

        print(f"Query: {name} ({len(seq)} aa, "
              f"{len(seq_to_kmers(seq, args.kmer_size))} unique {args.kmer_size}-mers)")
        if not hits:
            print("  No hits found.")
        else:
            print(f"  {'Rank':<6} {'Score':>8}  {'Length':>6}  Hit")
            print(f"  {'-'*5:<6} {'-'*8:>8}  {'-'*6:>6}  {'-'*50}")
            for rank, (score, db_header, db_len) in enumerate(hits, 1):
                # Truncate long headers
                display = db_header[:70] + "..." if len(db_header) > 70 else db_header
                print(f"  {rank:<6} {score:>8.4f}  {db_len:>6}  {display}")
        print()

    elapsed = t_search - t_start
    search_time = t_search - t_prep
    throughput = db_size_gb / search_time if search_time > 0 else 0

    print("=" * 70)
    print(f"Search complete — {db_size_gb:.1f} GB in {search_time:.1f}s "
          f"({throughput:.2f} GB/s)")
    print(f"Total time:  {elapsed:.1f}s  |  {num_cpus} CPUs, {num_nodes} node(s)")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    exit(main() or 0)
