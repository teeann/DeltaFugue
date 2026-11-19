import torch
import torch.nn as nn
import torch.nn.functional as F

# Ported from the JAX/Haiku `TapeInputLengthJumpCore` in
# https://github.com/google-deepmind/neural_networks_chomsky_hierarchy/blob/main/models/tape_rnn.py
#
# The tape never moves a pointer over a fixed array. Instead the whole tape is
# permuted every step so that index 0 is always the "head" position: writing
# always overwrites index 0, and the 5 actions (write+stay, write+left,
# write+right, write+jump_left, write+jump_right) are circular shifts of the
# tape applied via a soft (softmax-weighted) mixture of the 5 shifted
# candidates, which is what makes the whole thing differentiable without a
# discrete argmax.
#
# The two "jump" actions shift by `input_length` -- in the original JAX code
# this is a single Python int shared by the whole batch, since that repo's
# datasets generate every sequence in a batch at the same fixed length. This
# repo's `FormalLanguageDataset` samples each example's length independently
# (via `generate_sample(min_length, max_length, seed)` per-index) and pads to
# a shared batch width, so different rows of the same batch need *different*
# jump amounts -- a single scalar `torch.roll` shared across the batch isn't
# enough. `_shifted_gather` below takes a per-example shift tensor and gathers
# accordingly, so `write_stay`/`write_left`/`write_right` broadcast a constant
# shift (0/-1/+1) while `write_jump_left`/`write_jump_right` use a per-row
# `input_length` tensor.
#
# `input_length` itself has to be recovered from the raw token ids: the
# shared training loop in train.py only ever calls `model(x)` (see
# `models/stack_rnn.py`/`models/lstm.py`'s forward signatures), never passing
# the sequence mask, so it can't be read off the mask directly. Every task in
# `data_dir/fl_tasks/` reserves token id 0 for padding/dummy positions and
# never emits it as real content (see `duplicate_string.py`'s `PAD_IDX = 0`
# convention and `parity.py`'s char_to_index mapping, which starts at 1) and
# `collate_fn` also pads with 0, so counting the non-zero prefix of `x` gives
# a task-agnostic *starting point*: for `parity` (no interior padding) it's
# already exactly the true sequence length.
#
# For `duplicate_string` ([string] + [SEP] + [dummy]*len(string)) the correct
# jump target is `len(string)` alone, NOT the non-zero count of
# `len(string) + 1` (which also counts the SEP token) -- tracing the head's
# absolute tape address confirms this: writing `string[t]` while advancing the
# head by +1 each of the `len(string)` string steps leaves the head at
# absolute address `len(string)` by the time SEP is the input token; a single
# jump right then has to move the head back by exactly `len(string)` (not
# `len(string) + 1`) to land back on address 0, where `string[0]` was written,
# so the first dummy/output step reads the right value. Getting this wrong by
# +1 measurably matters: with the off-by-one jump the model plateaus near
# chance accuracy on `duplicate_string` (verified) instead of learning to
# copy. Since the SEP token is a single extra non-zero token specific to this
# task's framing (not something derivable task-agnostically from `x` alone
# without assuming a delimiter convention -- e.g. `parity` has no delimiter,
# and a fixed "last vocab id = delimiter" rule would incorrectly subtract real
# repeated content tokens for tasks like `parity` where the last id is a
# legitimate, frequently-repeated symbol), `input_length_offset` is an
# explicit, per-task-configured constant subtracted from the non-zero count
# rather than an attempt to infer delimiter-ness from `x` itself.


def _shifted_gather(memory: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """Circularly shifts `memory` along dim=1 by a per-row amount.

    memory: (B, M, C). shift: (B,) int64, positive or negative, arbitrary
    magnitude (wrapped mod M). Matches `jnp.roll(a, shift)[i] = a[(i - shift) % M]`.
    """
    B, M, C = memory.shape
    idx = torch.arange(M, device=memory.device).unsqueeze(0)  # (1, M)
    idx = (idx - shift.unsqueeze(1)) % M  # (B, M)
    idx = idx.unsqueeze(-1).expand(-1, -1, C)
    return torch.gather(memory, dim=1, index=idx)


class TapeRNNCell(nn.Module):
    """A single time-step of a Multi-Layer Tape RNN using nn.RNN."""

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        memory_cell_size,
        n_tapes=1,
        memory_size=30,
        mlp_hidden_sizes=(64, 64),
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_cell_size = memory_cell_size
        self.n_tapes = n_tapes
        self.memory_size = memory_size
        self.num_actions = 5  # stay, left, right, jump_left, jump_right

        self.rnn = nn.RNN(
            input_size=input_size + (n_tapes * memory_cell_size),
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        mlp_layers = []
        in_dim = hidden_size
        for hidden_dim in mlp_hidden_sizes:
            mlp_layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
            in_dim = hidden_dim
        mlp_layers.append(nn.Linear(in_dim, n_tapes * memory_cell_size))
        self.write_value_mlp = nn.Sequential(*mlp_layers)

        self.action_head = nn.Linear(hidden_size, n_tapes * self.num_actions)

    def _update_memory(self, memory, actions, write_values, input_length):
        """
        memory: (B, n_tapes, memory_size, memory_cell_size)
        actions: (B, n_tapes, num_actions) softmax probabilities
        write_values: (B, n_tapes, memory_cell_size)
        input_length: (B,) int64, shared across tapes
        """
        batch_size = memory.size(0)
        new_tapes = []
        for tape_idx in range(self.n_tapes):
            tape_memory = memory[:, tape_idx]  # (B, memory_size, cell)
            tape_write = write_values[:, tape_idx]  # (B, cell)
            memory_with_write = torch.cat(
                [tape_write.unsqueeze(1), tape_memory[:, 1:]], dim=1
            )

            zero_shift = torch.zeros(batch_size, dtype=torch.long, device=memory.device)
            one_shift = torch.ones(batch_size, dtype=torch.long, device=memory.device)
            candidates = torch.stack(
                [
                    _shifted_gather(memory_with_write, zero_shift),  # stay
                    _shifted_gather(memory_with_write, -one_shift),  # left
                    _shifted_gather(memory_with_write, one_shift),  # right
                    _shifted_gather(memory_with_write, -input_length),  # jump_left
                    _shifted_gather(memory_with_write, input_length),  # jump_right
                ],
                dim=1,
            )  # (B, num_actions, memory_size, cell)

            tape_actions = actions[:, tape_idx]  # (B, num_actions)
            new_tape = torch.einsum("banc,ba->bnc", candidates, tape_actions)
            new_tapes.append(new_tape)

        return torch.stack(new_tapes, dim=1)

    def forward(self, input_step, prev_state, input_length):
        """
        input_step: (B, input_dim)
        prev_state: (h, memory)
            h: (num_layers, B, hidden_size)
            memory: (B, n_tapes, memory_size, memory_cell_size)
        input_length: (B,) int64
        """
        h_prev, memory_prev = prev_state
        batch_size = input_step.size(0)

        current_memories = memory_prev[:, :, 0, :].reshape(
            batch_size, self.n_tapes * self.memory_cell_size
        )
        rnn_input = torch.cat([input_step, current_memories], dim=1).unsqueeze(1)

        output, h_new = self.rnn(rnn_input, h_prev)
        final_hidden = output.squeeze(1)

        write_values = self.write_value_mlp(final_hidden).view(
            batch_size, self.n_tapes, self.memory_cell_size
        )
        action_probs = F.softmax(
            self.action_head(final_hidden).view(batch_size, self.n_tapes, self.num_actions),
            dim=-1,
        )

        memory_curr = self._update_memory(memory_prev, action_probs, write_values, input_length)

        return h_new, memory_curr


class TapeRNN(nn.Module):
    def __init__(
        self,
        num_blocks,
        data_dim,
        model_dim,
        label_dim,
        dropout_rate,
        second_embedding=False,
        tape_size=30,
        n_tapes=1,
        mlp_hidden_sizes=(64, 64),
        input_length_offset=0,
    ):
        super().__init__()

        self.second_embedding = second_embedding
        self.model_dim = model_dim
        self.num_blocks = num_blocks
        self.n_tapes = n_tapes
        self.tape_size = tape_size
        self.orig_tape_size = tape_size
        self.memory_cell_size = 8
        self.input_length_offset = input_length_offset

        embedding_dim = model_dim // 2 if second_embedding else model_dim
        self.embedding = nn.Embedding(data_dim, embedding_dim)
        if second_embedding:
            self.embedding2 = nn.Embedding(data_dim, embedding_dim)

        self.cell = TapeRNNCell(
            input_size=model_dim,
            hidden_size=model_dim,
            num_layers=num_blocks,
            memory_cell_size=self.memory_cell_size,
            n_tapes=n_tapes,
            memory_size=tape_size,
            mlp_hidden_sizes=mlp_hidden_sizes,
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(model_dim, label_dim)

    def set_tape_size(self, new_size: int):
        self.orig_tape_size = self.tape_size
        self.tape_size = new_size
        self.cell.memory_size = new_size

    def reset_tape_size(self):
        self.tape_size = self.orig_tape_size
        self.cell.memory_size = self.orig_tape_size

    def mask_grads(self):
        pass

    def forward(self, x):
        # Token id 0 is reserved for padding/dummy positions across every
        # `data_dir/fl_tasks` task, so the non-zero prefix length of the raw
        # ids gives a task-agnostic starting point before any embedding
        # happens. `input_length_offset` (0 by default, e.g. 1 for
        # `duplicate_string`'s single SEP token) corrects for non-padding
        # delimiter tokens this count would otherwise wrongly include -- see
        # the module docstring above for the derivation.
        input_length = (x != 0).sum(dim=1).long() - self.input_length_offset

        if not self.second_embedding:
            emb = self.embedding(x)
        else:
            emb = torch.cat(
                [self.embedding(x[:, :, 0]), self.embedding2(x[:, :, 1])], dim=-1
            )

        batch_size, seq_len, _ = emb.size()

        h = torch.zeros(self.num_blocks, batch_size, self.model_dim, device=x.device)
        memory = torch.zeros(
            batch_size,
            self.n_tapes,
            self.tape_size,
            self.memory_cell_size,
            device=x.device,
        )

        outputs = []
        for t in range(seq_len):
            input_t = self.dropout(emb[:, t, :])
            h, memory = self.cell(input_t, (h, memory), input_length)
            outputs.append(h[-1])

        output_seq = torch.stack(outputs, dim=1)
        return self.linear(output_seq)
