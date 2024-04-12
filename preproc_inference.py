import torch
import pickle


def read_fasta(fasta_file):
    """Reads a FASTA file and returns a dictionary mapping sequence IDs to sequences."""
    seqs = {}
    with open(fasta_file, "r") as f:
        last_id = ""
        for line in f:
            if line.startswith(">"):
                last_id = line.strip()[1:]  # Remove the '>'
                seqs[last_id] = ""
            else:
                seqs[last_id] += line.strip()
    return seqs


def create_integer_encoding(seqs, seq_len=3000):
    """Creates an integer encoding for sequences."""
    X = torch.zeros(len(seqs), seq_len, dtype=torch.int)
    for idx, (seq_id, seq) in enumerate(seqs.items()):
        for col, nucl in enumerate(seq[:seq_len]):
            if nucl == "A":
                X[idx, col] = 1
            elif nucl == "C":
                X[idx, col] = 2
            elif nucl == "G":
                X[idx, col] = 3
            elif nucl == "T":
                X[idx, col] = 4
            # Non-ACGT characters remain as zero
    return X


def create_maps(seqs):
    """Creates mapping dictionaries for sequence IDs and rows."""
    map_seqid_to_row = {}
    map_row_to_seqid = {}
    for idx, seq_id in enumerate(seqs.keys()):
        map_seqid_to_row[seq_id] = idx
        map_row_to_seqid[idx] = seq_id
    return map_seqid_to_row, map_row_to_seqid


def preprocess_and_save(fasta_file, data_dir, seq_len=3000):
    """Reads FASTA, processes data, and saves required files for inference."""
    seqs = read_fasta(fasta_file)
    X = create_integer_encoding(seqs, seq_len)
    map_seqid_to_row, map_row_to_seqid = create_maps(seqs)
    torch.save(X, f"{data_dir}/seqs_tensor.pt")
    with open(f"{data_dir}/map_seqid_to_row.pkl", "wb") as f:
        pickle.dump(map_seqid_to_row, f)
    with open(f"{data_dir}/map_row_to_seqid.pkl", "wb") as f:
        pickle.dump(map_row_to_seqid, f)


# Usage
fasta_file = "your_fasta_file.fasta"
data_dir = "your_data_directory"
preprocess_and_save(fasta_file, data_dir)
