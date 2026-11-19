import torch
import warnings

# Vocab: 0=PAD/IGNORE, 1='0', 2='1', 3='*', 4='<SEP>'
vocab_size = 5
PAD_IDX = 0
SEP_IDX = 4

def binary_multiply(sequence):
    """
    Extracts 2 operands from a little-endian sequence and returns 
    their product as a little-endian list of tokens.
    """
    op_idx = sequence.index(3)
    
    # 1. Extract, reverse (little to big-endian), and shift tokens (1,2) back to bits (0,1)
    num1_bin = "".join(str(x - 1) for x in reversed(sequence[:op_idx]))
    num2_bin = "".join(str(x - 1) for x in reversed(sequence[op_idx + 1:]))
    
    # 2. Multiply them natively
    prod_val = int(num1_bin, 2) * int(num2_bin, 2)
    
    # 3. Convert product back to binary, reverse to little-endian, shift back to tokens
    if prod_val == 0:
        solution = [1] # Token 1 is '0'
    else:
        prod_bin = bin(prod_val)[2:]
        solution = [int(bit) + 1 for bit in reversed(prod_bin)]
    
    return solution

def generate_sample(min_length, max_length, seed=None):
    """
    Generates a sample for N x 3 Binary (Little-Endian) Multiplication.
    Operand 1 length is bounded by [min_length, max_length].
    Operand 2 length is strictly 3.
    """
    if seed is not None:
        torch.manual_seed(seed)

    if min_length > max_length:
        raise ValueError("min_length must be less than or equal to max_length")

    # 1. Generate Operand 1 (Length N)
    len1 = torch.randint(min_length, max_length - 3, (1,)).item()
    op1 = [torch.randint(1, 3, (1,)).item() for _ in range(len1)]
    
    # Prevent leading zeros in math (which are trailing in little-endian)
    op1[-1] = 2 
    
    # 2. Generate Operand 2 (Strictly Length 3)
    # --> This is the only line that changes <--
    op2 = [torch.randint(1, 3, (1,)).item() for _ in range(3)] 
    
    # 3. Stitch together: [Op1] + [*] + [Op2]
    sequence = op1 + [3] + op2
    
    # 4. Compute the true solution
    solution = binary_multiply(sequence)
    
    return sequence, solution

def preprocess_data(sample):
    """
    Preprocess function with Prompt + Dummy Token logic (Masked Prediction).
    Input:  [Op1, *, Op2] + [SEP] + [DUMMY] * len(Result)
    Target: [IGNORE] * len(Prompt) + [Result]
    """
    input_list, target_list = sample

    num_dummies = len(target_list)

    # 1. Construct Input: [Op1, *, Op2] + [SEP] + [DUMMY] * len(Result)
    input_seq = input_list + [SEP_IDX] + [PAD_IDX] * num_dummies

    # 2. Construct Target: [IGNORE] * len(Prompt) + [Result]
    prompt_len = len(input_list) + 1  # +1 for SEP
    target_seq = [PAD_IDX] * prompt_len + target_list

    # 3. Construct Mask
    mask = torch.zeros(len(input_seq), dtype=torch.bool)
    mask[prompt_len:] = True

    return (
        torch.tensor(input_seq, dtype=torch.long),
        torch.tensor(target_seq, dtype=torch.long),
        mask,
    )

def decode_tokens(token_list):
    """Helper to visualize tokens as strings for debugging."""
    mapping = {0: "[PAD]", 1: "0", 2: "1", 3: "*", 4: "<SEP>"}
    if isinstance(token_list, torch.Tensor):
        token_list = token_list.tolist()
    return " ".join(mapping.get(t, "?") for t in token_list)


if __name__ == "__main__":
    print("=== Masked Prediction N x 2 Binary Multiplication ===\n")

    for i in range(2):
        print(f"--- Sample {i+1} ---")

        raw_seq, raw_tgt = generate_sample(min_length=3, max_length=8)
        print(f"Human Readable: {decode_tokens(raw_seq)} \t -> \t {decode_tokens(raw_tgt)}")

        inp, tgt, msk = preprocess_data((raw_seq, raw_tgt))

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
