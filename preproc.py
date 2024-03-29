"""
Proprocesses molecular data for use in pytorch contrastive learning model.

INPUT: 
    a. Fasta file with all sequences. Sequence name is of format >{subtype}.{sequence_id}
    b. Subtype distance matrix (csv file)

0. Perform train test split
1. Create integer encoding tensor 
2. Create the following dictionaries for both train and test sets:
    a. map_seqid_to_row: maps sequence id to row number in integer encoding tensor
    b. map_row_to_seqid: maps row number in integer encoding tensor to sequence id
    c. map_subtype_to_seqids: maps subtype to list of sequence ids
    d. map_seqid_to_subtype: maps sequence id to subtype
    e. map_seqid_to_sequence: maps sequence id to sequence
3. Create contrast set for each sequence
    1. Find similar sequences (same subtype)
    2. Find dissimilar sequences (different subtype)
        include both the closest and farther subtypes which are not from the same subtype
    Set size = 5 similar + 5 close dissimilar + 5 random dissimilar = 15 sequences per contrast set

"""


import torch
from sklearn.model_selection import train_test_split as sklearn_train_test_split
import pandas as pd
from tqdm import tqdm
import pickle
import random

# input parameters

fasta_file = r"C:\Users\Dan\Desktop\CDC\Projects\dives\data\preproc\augmented_sequences.fasta"
metadata_file = r"C:\Users\Dan\Desktop\CDC\Projects\dives\data\preproc\subtypes.csv"
output_dir = r"C:\Users\Dan\Desktop\CDC\Projects\dives\data\preproc"


# Global variables
SEQ_LEN = 3000
RAND_SEED = 42
random.seed(RAND_SEED)
meta = pd.read_csv(metadata_file, encoding="ISO-8859-1")

# we only use meta to map sequence ids to subtypes, so let's just read that map as a dict, where the key is the sequence id and the value is the subtype
get_subtype_dict = {seqid: subtype for seqid, subtype in zip(meta["id"].values, meta["subtype"].values)}

def get_subtype(seqid):
    subtype, sid = seqid.split(".")
    if "spawned" in sid:
        return get_subtype_dict[sid.split("_")[0]]
    return get_subtype_dict[sid]


def main():
    print("Reading fasta file...")
    seqs_dictionary = read_fasta(fasta_file)
    print("Reading distance matrix file...")
    print("Splitting into train and test sets...")
    train_seqs, test_seqs = strat_train_test_split(seqs_dictionary)

    (
        train_map_seqid_to_row,
        train_map_row_to_seqid,
        train_map_subtype_to_seqids,
        train_map_seqid_to_subtype,
        test_map_seqid_to_row,
        test_map_row_to_seqid,
        test_map_subtype_to_seqids,
        test_map_seqid_to_subtype,
        map_label_to_subtype,
        map_subtype_to_label,
    ) = create_maps(train_seqs, test_seqs)

    train_seqs_tensor = create_integer_encoding(train_seqs)
    save((train_seqs_tensor, "train_seqs_tensor"))
    test_seqs_tensor = create_integer_encoding(test_seqs)
    save((test_seqs_tensor, "test_seqs_tensor"))

    # train_seqs_tensor = load("train_seqs_tensor.pt")
    # test_seqs_tensor = load("test_seqs_tensor.pt")

    train_labels_tensor = create_labels_tensor(train_seqs, map_subtype_to_label)
    test_labels_tensor = create_labels_tensor(test_seqs, map_subtype_to_label)


    save(
        (train_seqs_tensor, "train_seqs_tensor"),
        (test_seqs_tensor, "test_seqs_tensor"),
        (train_labels_tensor, "train_labels_tensor"),
        (test_labels_tensor, "test_labels_tensor"),
        (train_map_seqid_to_row, "train_map_seqid_to_row"),
        (train_map_row_to_seqid, "train_map_row_to_seqid"),
        (train_map_subtype_to_seqids, "train_map_subtype_to_seqids"),
        (train_map_seqid_to_subtype, "train_map_seqid_to_subtype"),
        (test_map_seqid_to_row, "test_map_seqid_to_row"),
        (test_map_row_to_seqid, "test_map_row_to_seqid"),
        (test_map_subtype_to_seqids, "test_map_subtype_to_seqids"),
        (test_map_seqid_to_subtype, "test_map_seqid_to_subtype"),
        (map_label_to_subtype, "map_label_to_subtype"),
        (map_subtype_to_label, "map_subtype_to_label"),
    )


def read_fasta(fasta_file):
    # global SEQ_LEN

    seqs = {}
    max_len = 0

    with open(fasta_file, "r") as f:
        last_id = ""
        for i, line in enumerate(f):
            if i % 2 == 0:
                last_id = line.strip()[1:]  # removing ">"
                seqs[last_id] = ""
            else:
                seqs[last_id] = line.strip()
                if len(line.strip()) > max_len:
                    max_len = len(line.strip())

    # SEQ_LEN = max_len
    return seqs


def get_consensus(seqs):
    # seqs is a list of sequences
    # returns the consensus sequence
    consensus = ""
    for i in range(SEQ_LEN):
        counts = {}
        for seq in seqs:
            if len(seq) <= i:
                continue
            if seq[i] == "-":
                continue
            if seq[i] not in counts:
                counts[seq[i]] = 0
            counts[seq[i]] += 1
        if counts == {}:
            consensus += "-"
        else:
            consensus += max(counts, key=counts.get)
    return consensus
            


def train_test_split(seqs_dict):
    seqs = list(seqs_dict.keys())
    random.shuffle(seqs)
    # train_seqs, test_seqs = sklearn_train_test_split(seqs, test_size=0.2, random_state=RAND_SEED)
    train_seqs, test_seqs = sklearn_train_test_split(seqs, test_size=0.2, random_state=RAND_SEED)
    train_seqs_dict = {seq: seqs_dict[seq] for seq in train_seqs}
    test_seqs_dict = {seq: seqs_dict[seq] for seq in test_seqs}
    return train_seqs_dict, test_seqs_dict


def strat_train_test_split(seqs_dict):
    # stratified train test split based on subtypes
    seqs = list(seqs_dict.keys())
    subtypes = [get_subtype(seq) for seq in seqs]
    train_seqs, test_seqs = sklearn_train_test_split(seqs, test_size=0.2, stratify=subtypes, random_state=RAND_SEED)
    train_seqs_dict = {seq: seqs_dict[seq] for seq in train_seqs}
    test_seqs_dict = {seq: seqs_dict[seq] for seq in test_seqs}
    return train_seqs_dict, test_seqs_dict


def create_integer_encoding(seqs_dict):
    X = torch.zeros(len(seqs_dict), SEQ_LEN, dtype=torch.float)
    for row, (id, seq) in tqdm(enumerate(seqs_dict.items()), desc="Integer Encoding", total=len(seqs_dict.keys())):
        for col, nucl in enumerate(seq):
            nucl = nucl.upper()
            if nucl not in ["A", "C", "G", "T"]:
                X[row][col] = 0
            elif nucl == "A":
                X[row][col] = 1
            elif nucl == "C":
                X[row][col] = 2
            elif nucl == "G":
                X[row][col] = 3
            elif nucl == "T":
                X[row][col] = 4
    return X


def create_labels_tensor(seqs_dict, map_subtype_to_label):
    # create a vector of length N giving the integer label for each sequence
    labels = torch.zeros(len(seqs_dict))
    for row, (id, seq) in tqdm(enumerate(seqs_dict.items()), desc="Creating Labels Tensor", total=len(seqs_dict.keys())):
        subtype = get_subtype(id)
        labels[row] = map_subtype_to_label[subtype]
    return labels



def save(*args):
    # function to save an arbitrary length list of parameters by checking their type, if torch tensor use torch save else use pickle
    for arg, name in args:
        if type(arg) == torch.Tensor:
            torch.save(arg, f"{output_dir}/{name}.pt")
        else:
            with open(f"{output_dir}/{name}.pkl", "wb") as f:
                pickle.dump(arg, f)

def load(name):
    if name.endswith(".pt"):
        return torch.load(f"{output_dir}/{name}")
    else:
        with open(f"{output_dir}/{name}", "rb") as f:
            return pickle.load(f)

def create_maps(train_seqs, test_seqs):
    """
    creating these maps:
        1. train_map_seqid_to_row,
        2. train_map_row_to_seqid,
        3. train_map_subtype_to_seqids,
        4. train_map_seqid_to_subtype,
        5. test_map_seqid_to_row,
        6. test_map_row_to_seqid,
        7. test_map_subtype_to_seqids,
        8. test_map_seqid_to_subtype,
        9. map_label_to_subtype,
        10. map_subtype_to_label,
    """

    train_map_seqid_to_row = {}
    train_map_row_to_seqid = {}
    train_map_subtype_to_seqids = {}
    train_map_seqid_to_subtype = {}
    train_subtypes = []
    for row, (id, seq) in tqdm(enumerate(train_seqs.items()), desc="Creating Train Maps", total=len(train_seqs.keys())):
        train_map_seqid_to_row[id] = row
        train_map_row_to_seqid[row] = id
        # subtype = id.split(".")[0]
        subtype = get_subtype(id)
        train_map_seqid_to_subtype[id] = subtype
        if subtype not in train_map_subtype_to_seqids:
            train_map_subtype_to_seqids[subtype] = []
        train_map_subtype_to_seqids[subtype].append(id)
        train_subtypes.append(subtype)

    test_map_seqid_to_row = {}
    test_map_row_to_seqid = {}
    test_map_subtype_to_seqids = {}
    test_map_seqid_to_subtype = {}
    test_subtypes = []
    for row, (id, seq) in tqdm(enumerate(test_seqs.items()), desc="Creating Test Maps", total=len(test_seqs.keys())):
        test_map_seqid_to_row[id] = row
        test_map_row_to_seqid[row] = id
        subtype = get_subtype(id)
        test_map_seqid_to_subtype[id] = subtype
        if subtype not in test_map_subtype_to_seqids:
            test_map_subtype_to_seqids[subtype] = []
        test_map_subtype_to_seqids[subtype].append(id)
        test_subtypes.append(subtype)

    map_label_to_subtype = {}
    map_subtype_to_label = {}

    # for i, subtype in enumerate(set(train_subtypes + test_subtypes)):
    #     map_label_to_subtype[i] = subtype
    #     map_subtype_to_label[subtype] = i

    #i want to use the metadata file to ensure all subtypes are included in the label map
    for i, subtype in enumerate(meta["subtype"].unique()):
        map_label_to_subtype[i] = subtype
        map_subtype_to_label[subtype] = i

    return (
        train_map_seqid_to_row,
        train_map_row_to_seqid,
        train_map_subtype_to_seqids,
        train_map_seqid_to_subtype,
        test_map_seqid_to_row,
        test_map_row_to_seqid,
        test_map_subtype_to_seqids,
        test_map_seqid_to_subtype,
        map_label_to_subtype,
        map_subtype_to_label,
    )

if __name__ == "__main__":
    main()