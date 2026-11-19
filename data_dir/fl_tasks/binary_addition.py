import warnings
import torch

# Vocab: 0=PAD/DUMMY/IGNORE, 1='0', 2='1', 3='+', 4='<SEP>'
vocab_size = 5
PAD_IDX = 0
SEP_IDX = 4


def binary_add(sequence):
    """
    Extracts operands from a little-endian sequence and returns 
    their sum as a little-endian list of tokens.
    """
    op_idx = sequence.index(3)
    
    # 1. Extract, reverse (little to big-endian), and shift tokens (1,2) back to bits (0,1)
    num1_bin = "".join(str(x - 1) for x in reversed(sequence[:op_idx]))
    num2_bin = "".join(str(x - 1) for x in reversed(sequence[op_idx + 1:]))
    
    # 2. Add them natively
    sum_val = int(num1_bin, 2) + int(num2_bin, 2)
    
    # 3. Convert sum back to binary, reverse to little-endian, shift back to tokens
    sum_bin = bin(sum_val)[2:]
    solution = [int(bit) + 1 for bit in reversed(sum_bin)]
    
    return solution


def generate_sample(min_length, max_length, seed=None):
    """
    Generates a single raw sample for the Binary (little-Endian) Addition task.
    Returns: (raw_input_sequence, raw_target_solution)
    """
    if seed is not None:
        torch.manual_seed(seed)

    if min_length > max_length:
        raise ValueError("min_length must be less than or equal to max_length")

    if min_length < 5:
        min_length = 5
        warnings.warn("Sequence for binary addition must be at least 5. Adjusted to 5.")

    # 1. Generate total input length (Operand 1 + '+' + Operand 2)
    length = torch.randint(min_length, max_length + 1, (1,)).item()
    
    # 2. Generate random bits (tokens 1 or 2)
    sequence = [torch.randint(1, 3, (1,)).item() for _ in range(length)]
    
    # 3. Place the operator (+) randomly, creating the imbalance
    op_idx = torch.randint(1, length - 1, (1,)).item()
    sequence[op_idx] = 3
    
    # 4. Prevent leading zeros (which are trailing in our little-endian format)
    sequence[op_idx - 1] = 2
    sequence[-1] = 2
    
    # 5. Compute the true solution
    solution = binary_add(sequence)
    
    # Return purely the raw components. Padding/SEP logic moves to preprocess.
    return sequence, solution


def preprocess_data(sample):
    """
    Preprocess function with Prompt + Dummy Token logic (Masked Prediction).
    Input:  [Op1, +, Op2] + [SEP] + [DUMMY] * len(Result)
    Target: [IGNORE] * len(Prompt)        + [Result]
    """
    input_list, target_list = sample

    # 1. Calculate needed dummies
    num_dummies = len(target_list)

    # 2. Construct Input: [Op1, +, Op2] + [SEP] + [DUMMY] * len(Result)
    input_seq = input_list + [SEP_IDX] + [PAD_IDX] * num_dummies

    # 3. Construct Target: [IGNORE] * len(Prompt) + [Result]
    prompt_len = len(input_list) + 1  # +1 to account for the SEP token
    target_seq = [PAD_IDX] * prompt_len + target_list

    # 4. Construct Mask
    # 0 (False) for prompt, 1 (True) for result
    mask = torch.zeros(len(input_seq), dtype=torch.bool)
    mask[prompt_len:] = True

    return (
        torch.tensor(input_seq, dtype=torch.long),
        torch.tensor(target_seq, dtype=torch.long),
        mask,
    )


def decode_tokens(token_list):
    """Helper to visualize tokens as strings for debugging."""
    mapping = {0: "[DUMMY/IGNORE]", 1: "0", 2: "1", 3: "+", 4: "<SEP>"}
    return " ".join(mapping.get(t, "?") for t in token_list)


if __name__ == "__main__":
    print("=== Masked Prediction Binary Addition Debugging ===\n")

    for i in range(2):
        print(f"--- Sample {i+1} ---")

        # 1. Raw Generation
        raw_seq, raw_tgt = generate_sample(min_length=5, max_length=6)
        print(f"Human Readable: {decode_tokens(raw_seq)} \t -> \t {decode_tokens(raw_tgt)}")

        # 2. Preprocessing for Masked Loss
        inp, tgt, msk = preprocess_data((raw_seq, raw_tgt))

        print("\nSynchronous Alignment:")
        print(f"Input (x):  {inp.tolist()}")
        print(f"Target (y): {tgt.tolist()}")
        print(f"Mask:       {[int(m) for m in msk.tolist()]}")

        # 3. Visualizing the Synchronous Prediction
        print("\nStep-by-step synchronous prediction:")
        inp_list = inp.tolist()
        tgt_list = tgt.tolist()
        
        for j in range(len(inp_list)):
            current_input_token = inp_list[j]
            token_it_predicts = tgt_list[j]
            is_calculated_in_loss = msk[j].item()

            calc_str = "-> [LOSS]" if is_calculated_in_loss else "-> [IGNORE]"
            print(
                f"  Input '{decode_tokens([current_input_token]):>14}' \t predicts \t "
                f"'{decode_tokens([token_it_predicts]):>14}' \t {calc_str}"
            )
        print("\n" + "=" * 60 + "\n")
