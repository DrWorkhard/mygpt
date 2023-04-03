import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB_SIZE = 10+3 # all digits plus =, +, end_token
DIGITS: int = 5
MAX_NUMBER: int = 10**(DIGITS)-1
MAX_TRAIN_NUMBER: int = 10**(DIGITS-1)-1
TEST_PRIMES: list[int] = [5, 17, 93, 7691]

SENTENCE_LENGTH = DIGITS +1+ DIGITS +1+ DIGITS+1 # add two numbers, resulting number is potentially one longer, a + and a =
EMBEDDING_DIM = 256
FEED_FORWARD_DIM = 256
ATTENTION_HEADS = 8
ATTENTION_BLOCKS = 6
ATTENTION_EMBEDDING_DIM = EMBEDDING_DIM // ATTENTION_HEADS

EPOCHS = 30
STEPS_PER_EPOCH = 10**3
BATCH_SIZE = 10

def contains_prime(number):
    for p in TEST_PRIMES:
        if number % p == 0:
            return True
    return False

train_numbers: list[int] = range(0, MAX_TRAIN_NUMBER+1)
train_numbers = [n for n in train_numbers if not contains_prime(n)]

val_interpo_numbers: list[int] = range(0, MAX_TRAIN_NUMBER+1)
val_interpo_numbers = [n for n in val_interpo_numbers if contains_prime(n)]

val_extrapo_numbers: list[int] = list(range(MAX_TRAIN_NUMBER+1, MAX_NUMBER+1))

def get_batch(numbers, size):
    batch_numbers = np.random.choice(numbers, (size, 2), False)
    input_strings = []
    target_strings = []
    for s1, s2 in batch_numbers:
        in_s = f"{s1:0{DIGITS}d}+{s2:0{DIGITS}d}={s1+s2:0{DIGITS+1}d}"
        target_s = in_s[1:] + " " # (close the target string with an "end token")
        input_strings.append(in_s)
        target_strings.append(target_s)
    return input_strings, target_strings

mapping = {str(i): i for i in range(10)}
mapping['='] = 10
mapping['+'] = 11
mapping[' '] = 12
def tokenize(example: str):
    chars = [*example]
    tokens = [mapping[c] for c in chars]
    return tokens

reverse_mapping = {value: key for key, value in mapping.items()}
def detokenize(tokens: list[int]) -> str:
    chars = [reverse_mapping[t] for t in tokens]
    return "".join(chars)


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
        self.to_query = nn.Linear(EMBEDDING_DIM, ATTENTION_EMBEDDING_DIM*ATTENTION_HEADS, False)
        self.to_key = nn.Linear(EMBEDDING_DIM, ATTENTION_EMBEDDING_DIM*ATTENTION_HEADS, False)
        self.to_value = nn.Linear(EMBEDDING_DIM, ATTENTION_EMBEDDING_DIM*ATTENTION_HEADS, False)

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
        q = q.view(BATCH_SIZE, SENTENCE_LENGTH, ATTENTION_HEADS, ATTENTION_EMBEDDING_DIM).movedim(1,2)
        k = k.view(BATCH_SIZE, SENTENCE_LENGTH, ATTENTION_HEADS, ATTENTION_EMBEDDING_DIM).movedim(1,2)
        v = v.view(BATCH_SIZE, SENTENCE_LENGTH, ATTENTION_HEADS, ATTENTION_EMBEDDING_DIM).movedim(1,2)

        k = k.transpose(-2,-1)
        attention_matrix = torch.matmul(q, k) / np.sqrt(float(ATTENTION_EMBEDDING_DIM))
        attention_matrix.masked_fill(self.mask, -torch.inf)
        weights = F.softmax(attention_matrix, -1)

        x = torch.matmul(weights, v)

        # concat heads
        x = x.movedim(1,2).contiguous()
        x = x.view(BATCH_SIZE, SENTENCE_LENGTH, ATTENTION_HEADS*ATTENTION_EMBEDDING_DIM)
        return x


class CPT(nn.Module):
    
    def __init__(self):
       super().__init__()
       self.token_embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
       self.position_embeddding = nn.Embedding(SENTENCE_LENGTH, EMBEDDING_DIM)
       self.block = AttentionBlock()
       self.head = nn.Linear(EMBEDDING_DIM, VOCAB_SIZE) # TODO: weight tying


    def forward(self, examples: list[str], targets: list[str] = None):
        idx: list[list[int]] = [tokenize(example) for example in examples]

        idx_tensor = torch.tensor(idx) # batch_size x sentence_length
        position_tensor = torch.tensor(range(SENTENCE_LENGTH))

        pos = self.position_embeddding(position_tensor) # sentence_length x embeding_dimension
        x = self.token_embedding(idx_tensor) # batch_size x sentence_length x embedding_dimension
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
            expected_output = torch.tensor(expected_output_idx)
            expected_output_flat = expected_output.view(-1)
            loss = F.cross_entropy(log_softmax_flat, expected_output_flat, reduction='none')

        return out_probabilities, loss
        

cpt = CPT()
optimizer = torch.optim.AdamW(cpt.parameters())

for epoch in range(EPOCHS):
    mean_loss = 0.
    for _ in range(STEPS_PER_EPOCH):
        input, target = get_batch(train_numbers, BATCH_SIZE)
        probs, loss = cpt(input, target)

        # now, since we dont want to regress before the = sign: crop it like its hot
        loss = loss.view(BATCH_SIZE, SENTENCE_LENGTH)
        loss = loss[:,DIGITS*2+1:] # we evaluate from the = to the " " (which should always be fixed)
        loss = loss.mean()
        mean_loss +=loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    mean_loss = mean_loss / STEPS_PER_EPOCH
    print(f"epoch {epoch} loss: {mean_loss}")


NUM_INFERENCE_EXAMPLES = 10
input, _ = get_batch(val_interpo_numbers, NUM_INFERENCE_EXAMPLES)
for i in range(DIGITS*2+1, SENTENCE_LENGTH):
    probs, _ = cpt(input)



