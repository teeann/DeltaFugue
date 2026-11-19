import torch

# Vocab: 0=PAD/IGNORE, 1=Binary '0', 2=Binary '1', 3=MASK, 4=ODD_PAD, 5=SEP
vocab_size = 6
PAD_IDX = 0
MASK_IDX = 3
ODD_PAD_IDX = 4
SEP_IDX = 5


def generate_sample(min_length, max_length, seed=None):
    """
    Generates a single raw sample for the Missing Duplicate task.
    Note: min_length and max_length now define the length of the base word 'w'.
    Total sequence length before prompt formatting will be 2x.
    """
    if seed is not None:
        torch.manual_seed(seed)

    if min_length > max_length:
        raise ValueError("min_length must be less than or equal to max_length")

    # Generate the length of the base word 'w'
    length = torch.randint(min_length, max_length + 1, (1,)).item()

    # 1. Generate the base binary string of length `length`
    w = [torch.randint(1, 3, (1,)).item() for _ in range(length)]
    
    # 2. Duplicate it
    ww = w + w
    
    # 3. Mask exactly one token across the duplicated string
    mask_idx = torch.randint(0, len(ww), (1,)).item()
    missing_token = ww[mask_idx]
    ww[mask_idx] = MASK_IDX
    
    return ww, [missing_token]


def preprocess_data(sample):
    """
    Preprocess function with Prompt + Dummy Token logic.
    Input:  [Masked ww (padded)] + [SEP] + [DUMMY]
    Target: [IGNORE] * len(Prompt)       + [Missing Token]
    """
    input_list, target_list = sample

    num_dummies = len(target_list)

    # Construct Input
    input_seq = input_list + [SEP_IDX] + [PAD_IDX] * num_dummies

    # Construct Target
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
    mapping = {0: "[PAD]", 1: "0", 2: "1", 3: "_", 4: "[ODD_PAD]", 5: "<SEP>"}
    return " ".join(mapping.get(t, "?") for t in token_list)


if __name__ == "__main__":
    print("=== DeepMind JAX-Aligned Missing Duplicate ===\n")

    for i in range(4):
        # 1. Raw Generation
        # Testing a mix of even and odd lengths to verify the ODD_PAD logic
        raw_seq, raw_tgt = generate_sample(min_length=4, max_length=8, seed=42+i)
        print(f"--- Sample {i+1} ---")
        print(f"Masked String: {decode_tokens(raw_seq)} \t -> \t Target: {decode_tokens(raw_tgt)}")

        # 2. Preprocessing for Masked Loss
        inp, tgt, msk = preprocess_data((raw_seq, raw_tgt))

        print(f"Input (x):  {inp.tolist()}")
        print(f"Target (y): {tgt.tolist()}")
        print(f"Mask:       {[int(m) for m in msk.tolist()]}\n")
