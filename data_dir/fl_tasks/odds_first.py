import torch

# Vocab: 0=PAD/IGNORE, 1=Binary '0', 2=Binary '1', 3=SEP
vocab_size = 4
PAD_IDX = 0
SEP_IDX = 3

def generate_sample(min_length, max_length, seed=None):
    """
    Generates a single raw sample for the Odds First task.
    Returns: (raw_sequence, sorted_target)
    """
    if seed is not None:
        torch.manual_seed(seed)

    if min_length > max_length:
        raise ValueError("min_length must be less than or equal to max_length")

    # Generate the full requested length (unlike the old length // 2 logic)
    length = torch.randint(min_length, max_length + 1, (1,)).item()

    # 1. Generate the base binary string (tokens 1 and 2)
    sequence = [torch.randint(1, 3, (1,)).item() for _ in range(length)]
    
    # 2. Extract odds and evens
    # Note: 1-indexed "odds" (s_1, s_3, s_5) correspond to 0-indexed 0, 2, 4
    odd_index_tokens = sequence[0::2]   
    even_index_tokens = sequence[1::2]  
    
    # 3. Concatenate to form the true "Odds First" target
    target = odd_index_tokens + even_index_tokens

    return sequence, target

def preprocess_data(sample):
    """
    Preprocess function with Prompt + Dummy Token logic.
    Input:  [Sequence] + [SEP] + [DUMMY] * len(Sequence)
    Target: [IGNORE] * len(Prompt)     + [Target Sequence]
    """
    input_list, target_list = sample

    num_dummies = len(target_list)

    # Construct Input: [Sequence] + [SEP] + [DUMMY]
    input_seq = input_list + [SEP_IDX] + [PAD_IDX] * num_dummies

    # Construct Target: [IGNORE] * len(Prompt) + [Target Sequence]
    prompt_len = len(input_list) + 1  # +1 to account for the SEP token
    target_seq = [PAD_IDX] * prompt_len + target_list

    # Construct Mask: 0 (False) for prompt, 1 (True) for the answer
    mask = torch.zeros(len(input_seq), dtype=torch.bool)
    mask[prompt_len:] = True

    return (
        torch.tensor(input_seq, dtype=torch.long),
        torch.tensor(target_seq, dtype=torch.long),
        mask,
    )

def decode_tokens(token_list):
    """Helper to visualize tokens as strings for debugging."""
    mapping = {0: "[PAD]", 1: "0", 2: "1", 3: "<SEP>"}
    return " ".join(mapping.get(t, "?") for t in token_list)

if __name__ == "__main__":
    print("=== Odds First Debugging ===\n")

    for i in range(3):
        # Raw Generation (Testing varying lengths)
        raw_seq, raw_tgt = generate_sample(min_length=6, max_length=9, seed=42+i)
        
        print(f"--- Sample {i+1} ---")
        print(f"Sequence: {decode_tokens(raw_seq)}")
        print(f"Target:   {decode_tokens(raw_tgt)}\n")

        # Preprocessing for Masked Loss
        inp, tgt, msk = preprocess_data((raw_seq, raw_tgt))

        print(f"Input (x):  {inp.tolist()}")
        print(f"Target (y): {tgt.tolist()}")
        print(f"Mask:       {[int(m) for m in msk.tolist()]}\n")
