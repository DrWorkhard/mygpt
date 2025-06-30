import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pylab as plt

DEVICE = torch.device("mps")

VOCAB_SIZE = 10 + 3  # all digits plus =, +, end_token: ' '
DIGITS: int = 5
MAX_NUMBER: int = 10 ** (DIGITS) - 1
MAX_TRAIN_NUMBER: int = 10 ** (DIGITS - 1) - 1
VAL_PRIMES = [17, 93, 7691]
TEST_PRIMES: list[int] = [5]

SENTENCE_LENGTH = (
    DIGITS + 1 + DIGITS + 1 + DIGITS + 1
)  # add two numbers, resulting number is potentially one longer, a + and a =
EMBEDDING_DIM = 256
FEED_FORWARD_DIM = 256
ATTENTION_HEADS = 8
ATTENTION_BLOCKS = 6
ATTENTION_EMBEDDING_DIM = EMBEDDING_DIM // ATTENTION_HEADS

EPOCHS = 100
STEPS_PER_EPOCH = 10**3
MAX_STEPS = EPOCHS * STEPS_PER_EPOCH
WARMUP_STEPS = 2 * STEPS_PER_EPOCH
BATCH_SIZE = 100
MAX_LEARNING_RATE = 6e-4
MIN_LEARNING_RATE = 6e-5


def get_lr(step):
    # cosine learning rate. First ramps up linearily to max lr, then ramps down with a half cosine to min lr.
    if step < WARMUP_STEPS:
        return step / WARMUP_STEPS * MAX_LEARNING_RATE

    cos_interpolator = (step - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
    lr = (
        MIN_LEARNING_RATE
        + (1 + np.cos(np.pi * cos_interpolator))
        * (MAX_LEARNING_RATE - MIN_LEARNING_RATE)
        / 2.0
    )
    assert lr > MIN_LEARNING_RATE - 10**-8

    if step > MAX_STEPS:
        return MIN_LEARNING_RATE
    return lr


def contains_prime(number: int, primes: list[int]):
    # filters numbers which are divisible by a prime in the list.
    # used to create train, val and test sets
    for p in primes:
        if number % p == 0:
            return True
    return False


train_numbers: list[int] = range(0, MAX_TRAIN_NUMBER + 1)
train_numbers = [
    n for n in train_numbers if not contains_prime(n, TEST_PRIMES + VAL_PRIMES)
]

val_interpo_numbers: list[int] = range(0, MAX_TRAIN_NUMBER + 1)
val_interpo_numbers = [n for n in val_interpo_numbers if contains_prime(n, VAL_PRIMES)]

test_interpo_numbers: list[int] = range(0, MAX_TRAIN_NUMBER + 1)
test_interpo_numbers = [
    n for n in test_interpo_numbers if contains_prime(n, TEST_PRIMES)
]

test_extrapo_numbers: list[int] = list(range(MAX_TRAIN_NUMBER + 1, MAX_NUMBER + 1))


def get_batch(numbers, size):
    batch_numbers = np.random.choice(numbers, (size, 2), False)
    input_strings = []
    target_strings = []
    for s1, s2 in batch_numbers:
        in_s = f"{s1:0{DIGITS}d}+{s2:0{DIGITS}d}={s1+s2:0{DIGITS+1}d}"
        target_s = in_s[1:] + " "  # (close the target string with an "end token")
        input_strings.append(in_s)
        target_strings.append(target_s)
    return input_strings, target_strings


mapping = {str(i): i for i in range(10)}
mapping["="] = 10
mapping["+"] = 11
mapping[" "] = 12


def tokenize(example: str):
    chars = [*example]
    tokens = [mapping[c] for c in chars]
    return tokens


reverse_mapping = {value: key for key, value in mapping.items()}


def detokenize(tokens: list[int]) -> str:
    chars = [reverse_mapping[t] for t in tokens]
    return "".join(chars)


# show 10 examples of each: train, val interpo, val extrapo
def judge_results(numbers, howmany, model, show=True):
    network_inputs, target = get_batch(numbers, howmany)
    probs, _ = model(network_inputs)
    results = probs.argmax(-1)
    results = [detokenize(result.cpu().numpy()) for result in results]
    results = [result[DIGITS * 2 + 1 : -1] for result in results]
    checks = [
        network_input.endswith(result)
        for network_input, result in zip(network_inputs, results)
    ]
    correct = sum(checks)
    print(f"correct: {correct} out of {howmany}")
    if show:
        print("first 10 as demonstration: ")
        for i in range(10):
            network_input, result, correct = network_inputs[i], results[i], checks[i]
            color = "\033[92m" if correct else "\033[91m"
            print(
                f"{color}input with solution: {network_input} / result: {result} / correct: {correct}\033[0m"
            )
    return correct


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(EMBEDDING_DIM, FEED_FORWARD_DIM)
        self.linear2 = nn.Linear(FEED_FORWARD_DIM, EMBEDDING_DIM)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm((EMBEDDING_DIM,))
        self.attention = MultiHeadAttention()
        self.layer_norm2 = nn.LayerNorm((EMBEDDING_DIM,))
        self.feed_forward = FeedForward()

    def forward(self, x):
        x = self.layer_norm1(x)
        x = x + self.attention(x)
        x = self.layer_norm2(x)
        x = x + self.feed_forward(x)
        return x


class MultiHeadAttention(nn.Module):
    # innefficient attention block

    def __init__(self):
        super().__init__()
        self.to_query = nn.Linear(
            EMBEDDING_DIM, ATTENTION_EMBEDDING_DIM * ATTENTION_HEADS, False
        )
        self.to_key = nn.Linear(
            EMBEDDING_DIM, ATTENTION_EMBEDDING_DIM * ATTENTION_HEADS, False
        )
        self.to_value = nn.Linear(
            EMBEDDING_DIM, ATTENTION_EMBEDDING_DIM * ATTENTION_HEADS, False
        )

        mask = torch.ones((1, 1, SENTENCE_LENGTH, SENTENCE_LENGTH))
        mask = torch.triu(mask, 1)
        mask = mask == 1
        self.register_buffer("mask", mask)

    def forward(self, x):
        # karpathy uses transpose instead of movedim. Is this faster?
        q = self.to_query(x)
        k = self.to_key(x)
        v = self.to_value(x)

        # split up into heads
        q = q.view(
            -1, SENTENCE_LENGTH, ATTENTION_HEADS, ATTENTION_EMBEDDING_DIM
        ).movedim(1, 2)
        k = k.view(
            -1, SENTENCE_LENGTH, ATTENTION_HEADS, ATTENTION_EMBEDDING_DIM
        ).movedim(1, 2)
        v = v.view(
            -1, SENTENCE_LENGTH, ATTENTION_HEADS, ATTENTION_EMBEDDING_DIM
        ).movedim(1, 2)

        k = k.transpose(-2, -1)
        attention_matrix = torch.matmul(q, k) / np.sqrt(float(ATTENTION_EMBEDDING_DIM))
        attention_matrix = attention_matrix.masked_fill(self.mask, -torch.inf)
        weights = F.softmax(attention_matrix, -1)

        x = torch.matmul(weights, v)

        # concat heads
        x = x.movedim(1, 2).contiguous()
        x = x.view(-1, SENTENCE_LENGTH, ATTENTION_HEADS * ATTENTION_EMBEDDING_DIM)
        return x


class CPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        self.position_embeddding = nn.Embedding(SENTENCE_LENGTH, EMBEDDING_DIM)
        self.block = AttentionBlock()
        self.head = nn.Linear(EMBEDDING_DIM, VOCAB_SIZE)  # TODO: weight tying

    def forward(self, examples: list[str], targets: list[str] = None):
        idx: list[list[int]] = [tokenize(example) for example in examples]

        idx_tensor = torch.tensor(idx, device=DEVICE)  # batch_size x sentence_length
        position_tensor = torch.tensor(range(SENTENCE_LENGTH), device=DEVICE)

        pos = self.position_embeddding(
            position_tensor
        )  # sentence_length x embeding_dimension
        x = self.token_embedding(
            idx_tensor
        )  # batch_size x sentence_length x embedding_dimension
        x += pos

        for _ in range(ATTENTION_BLOCKS):
            # lets goooo attention
            x = self.block(x)

        x = self.head(x)

        out_probabilities = F.softmax(x, -1)
        loss = None

        if targets:
            log_softmax = F.log_softmax(x, -1)
            log_softmax_flat = log_softmax.view(-1, VOCAB_SIZE)
            expected_output_idx = [tokenize(target) for target in targets]
            expected_output = torch.tensor(expected_output_idx, device=DEVICE)
            expected_output_flat = expected_output.view(-1)

            all_params = torch.cat([param.view(-1) for param in self.parameters()])
            l1_norm = torch.mean(torch.abs(all_params))
            l1_weight = 20.0

            loss = (
                F.cross_entropy(
                    log_softmax_flat, expected_output_flat, reduction="none"
                )
                + l1_norm * l1_weight
            )

        return out_probabilities, loss


if __name__ == "__main__":
    cpt = CPT()
    cpt.to(DEVICE)
    optimizer = torch.optim.AdamW(cpt.parameters())

    max_correct = 0
    for epoch in range(EPOCHS):
        start = time.time()
        mean_loss = 0.0
        for step in range(STEPS_PER_EPOCH):
            input, target = get_batch(train_numbers, BATCH_SIZE)
            probs, loss = cpt(input, target)

            loss = loss.view(BATCH_SIZE, SENTENCE_LENGTH)
            regression_loss = loss.mean()

            logging_loss = loss[
                :, DIGITS * 2 + 1 :
            ]  # we log the loss from the = to the last " ". This loss should approach 0
            mean_loss += logging_loss.mean()

            optimizer.zero_grad()
            regression_loss.backward()

            lr = get_lr(epoch * step)
            for g in optimizer.param_groups:
                g["lr"] = lr

            optimizer.step()
        mean_loss = mean_loss / STEPS_PER_EPOCH
        print(f"epoch {epoch} loss: {mean_loss}")

        end = time.time()
        epoch_duration = end - start
        print(f"took {int(epoch_duration//60)}min {int(epoch_duration%60)}s")

        print("train", end=" ")
        judge_results(train_numbers, 1000, cpt, False)
        print("val", end=" ")
        correct = judge_results(val_interpo_numbers, 340, cpt, False)

        if epoch > 10 and correct > max_correct:
            print("saving model")
            torch.save(
                {"model": cpt.state_dict(), "optimizer": optimizer.state_dict()},
                "model_and_optimizer_08_04_2023_val_max_l1_10.ckpt",
            )
            max_correct = correct

    cpt.eval()

    NUM_INFERENCE_EXAMPLES = 10
    input, _ = get_batch(val_interpo_numbers, NUM_INFERENCE_EXAMPLES)
    for i in range(DIGITS * 2 + 1, SENTENCE_LENGTH):
        probs, _ = cpt(input)
