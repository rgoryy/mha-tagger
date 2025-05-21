import torch
from torch import tensor
import torch.nn as nn
from torch.nn import MultiheadAttention as MHAtten
from torch.nn.functional import normalize as norm
import re

learning_rate = 0.2
epochs = 50
word_re = re.compile("^(\\d+)\t([^\\t]+)\\t([^\\t]+)\\t([A-Z]+)\\t(.*)$")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device ", device)

def compare(pred, target):
    if pred.shape != target.shape:
        raise BaseException("Shape mismatch")
    match_count = 0
    for p, t in zip(pred, target):
        p_max = p.topk(1).indices[0]
        t_max = t.topk(1).indices[0]
        if p_max == t_max:
            match_count = match_count + 1
    return float(match_count) / len(target)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, file_name, max_word_length=11):
        self.char_set = "щ<.ЪмцЛкб>|яй3:В;]\"ГУлБжэс+7_Ы!ФmНг-ды'ТИ(уеЁЭпит98o2[ЕМЩЦЖёх4ъА—ЬШeрЮ)КС65ЙЯзчш0юф?ОньДХ,аПЧР1оЗв/"
        self.char_dict = ['<pad>', '<unk>']
        self.char_dict_map = {'<pad>': 0, '<unk>': 1}

        for c in self.char_set:
            if c not in self.char_dict_map:
                self.char_dict.append(c.lower())
                self.char_dict_map[c.lower()] = (len(self.char_dict) - 1)

        print(self.char_dict_map)
        self.pos_dict = []
        self.pos_dict_map = dict()
        self.sents = []
        self.words = []
        self.max_word_length = max_word_length

        with open(file_name, encoding="utf-8") as f:
            lines = f.readlines()

        sent = None
        words = []
        for l in lines:
            if l.startswith("# text ="):
                if sent is not None:
                    self.sents.append(sent)
                    self.words.append(words)
                sent = l[len("# text = "):]
                words = []
                continue
            if l.startswith("#"):
                continue
            r = word_re.match(l)
            if not r:
                continue
            word = r.group(2)
            lemma = r.group(3)
            pos = r.group(4)

            if not pos in self.pos_dict_map.keys():
                self.pos_dict.append(pos)
                self.pos_dict_map[pos] = len(self.pos_dict) - 1

            words.append({"word": word.lower(), "pos": self.pos_dict_map[pos]})
            assert len(words) == int(r.group(1)), str(len(words)) + " not equals " + r.group(1)
        assert len(self.sents) == len(self.words)

    def word_to_tensor(self, word):
        indices = []
        for c in word.lower() [:max_word_length]:
            if c in self.char_dict_map:
                indices.append(self.char_dict_map[c])
            else:
                indices.append(1)

        indices += [0] * (self.max_word_length - len(indices))
        return torch.tensor(indices, dtype=torch.long)

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, index):
        sent = self.words[index]

        input = torch.zeros((len(sent), self.max_word_length), dtype=torch.long)
        for token_index, token in enumerate(sent):
            input[token_index] = self.word_to_tensor(token["word"])

        output = torch.zeros((len(sent), len(self.pos_dict)))
        for token_index, token in enumerate(sent):
            output[token_index][token["pos"]] = 1

        return (input, output)


class CharFF(nn.Module):
    def __init__(self, char_vocab_size, char_embed_dim, output_dim, dropout=0.2, hidden_dim=512):
        super().__init__()
        self.char_embedding = nn.Embedding(char_vocab_size, char_embed_dim)
        self.dense1 = nn.Linear(char_embed_dim * max_word_length, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.char_embedding(x)
        flattened = embedded.view(embedded.size(0), -1)

        chars_embedding = self.dropout(flattened)
        chars_embedding = self.relu(self.dense1(chars_embedding))
        chars_embedding = self.dropout(chars_embedding)
        chars_embedding = self.dense2(chars_embedding)
        chars_embedding = self.dropout(chars_embedding)

        return chars_embedding


class Model(nn.Module):
    def __init__(self, num_chars, char_embed_dim, word_embed_dim, atten_dim, num_pos, num_heads, max_word_length):
        super().__init__()
        self.char_ff = CharFF(num_chars, char_embed_dim, word_embed_dim, dropout=0.2, hidden_dim=512)
        self.ln_query = nn.Linear(word_embed_dim, atten_dim)
        self.ln_key = nn.Linear(word_embed_dim, atten_dim)
        self.ln_value = nn.Linear(word_embed_dim, atten_dim)
        self.atten = MHAtten(atten_dim, num_heads)
        self.ln_pos = nn.Linear(atten_dim, num_pos)

    def forward(self, x):
        e = self.char_ff(x)
        q = self.ln_query(e)
        k = self.ln_key(e)
        v = self.ln_value(e)
        atten, atten_weight = self.atten(q, k, v)
        return norm(self.ln_pos(atten), p=1, dim=1)


def train(m, train_set, loss_fn, opt):
    step = 0
    matching_sum = 0.0
    for input, target in train_set:
        pred = m(input.to(device))
        loss = loss_fn(pred, target.to(device))
        matching = compare(pred.to("cpu"), target) * 100
        matching_sum = matching_sum + matching
        loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 3)
        opt.step()
        opt.zero_grad()
        step = step + 1
    return matching_sum / step


data = "sent"

max_word_length = 11
char_embed_dim = 32
word_embed_dim = 1024
atten_dim = 1024
num_heads = 32

d = Dataset(data, max_word_length=max_word_length)
train_set, test_set = torch.utils.data.random_split(d, [.85, .15],
                                                    generator=torch.Generator(device="cpu").manual_seed(2024))


m = Model(len(d.char_dict), char_embed_dim, word_embed_dim,
          atten_dim, len(d.pos_dict), num_heads, max_word_length).to(device)
loss_fn = nn.MSELoss()
opt = torch.optim.SGD(m.parameters(), lr=learning_rate)

for i in range(epochs):
    print("Epoch ", i)
    print(train(m, train_set, loss_fn, opt))
