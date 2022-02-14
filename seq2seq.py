import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *
from data_util import batch2TrainData, normalizeString, indexesFromSentence


class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # 初始化GRU，这里输入和hidden大小都是hidden_size，因为我们这里假设embedding层的输出大小是hidden_size
        # 如果只有一层，那么不进行Dropout，否则使用传入的参数dropout进行GRU的Dropout
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        # 输入是(max_length, batch)，Embedding之后变成(max_length, batch, hidden_size)
        embedded = self.embedding(input_seq)
        # Pack padded batch of sequences for RNN module
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths.cpu())
        # 通过GRU进行forward计算，需要传入输入和隐变量
        outputs, hidden = self.gru(packed, hidden)

        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        # 返回最终的输出和最后时刻的隐状态。
        return outputs, hidden


# Luong 注意力layer
class Attn(torch.nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = torch.nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = torch.nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)


    def forward(self, hidden, encoder_outputs):

        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            # 计算内积
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()


        return F.softmax(attn_energies, dim=1).unsqueeze(1)

class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        # 保存到self里，attn_model就是前面定义的Attn类的对象。
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        # 定义Decoder的layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # NOTE：decoder每一步只能处理一个时刻的数据，因为t时刻计算完了才能计算t+1时刻

        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)

        rnn_output, hidden = self.gru(embedded, last_hidden)
        # 计算注意力权重， 根据前面的分析，attn_weights的shape是(64, 1, 10)
        attn_weights = self.attn(rnn_output, encoder_outputs)


        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # 把context向量和GRU的输出拼接起来
        # rnn_output从(1, 64, 500)变成(64, 500)
        rnn_output = rnn_output.squeeze(0)
        # context从(64, 1, 500)变成(64, 500)
        context = context.squeeze(1)
        # 拼接得到(64, 1000)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))

        # out是(500, 词典大小=7826)
        output = self.out(concat_output)
        # 用softmax变成概率，表示当前时刻输出每个词的概率。
        output = F.softmax(output, dim=1)
        # 返回 output和新的隐状态
        return output, hidden

# Masked loss

def maskNLLLoss(inp, target, mask):
    # 计算实际的词的个数，因为padding是0，非padding是1，因此sum就可以得到词的个数
    nTotal = mask.sum()
    # torch.gather函数首先把0.4和0.3（正确分类对应的概率值）选出来
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()


def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding,
          encoder_optimizer, decoder_optimizer, batch_size, clip, max_length=MAX_LENGTH):

    # 梯度清空
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # 设置device，从而支持GPU，当然如果没有GPU也能工作。
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)

    # 初始化变量
    loss = 0
    print_losses = []
    n_totals = 0

    # encoder的Forward计算
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Decoder的初始输入是SOS，我们需要构造(1, batch)的输入，表示第一个时刻batch个输入
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)


    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # 确定是否teacher forcing
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # 一次处理一个时刻
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # Teacher forcing: 下一个时刻的输入是当前正确答案
            decoder_input = target_variable[t].view(1, -1)
            # 计算累计的loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            # 计算累计的loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    # 反向计算
    loss.backward()

    # 梯度裁剪
    _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # 更新参数
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals

def evaluate(encoder, decoder, searcher, voc, sentence, max_length=MAX_LENGTH):
    ### 把输入的一个batch句子变成id
    indexes_batch = [indexesFromSentence(voc, sentence)]
    # 创建lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # 转置
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # 放到合适的设备上(比如GPU)
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # 用searcher解码
    tokens, scores = searcher(input_batch, lengths, max_length)
    # ID变成词。
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


def evaluateInput(encoder, decoder, searcher, voc):
    input_sentence = ''
    while(1):
        try:
            # 得到用户终端的输入
            input_sentence = input('> ')
            # 是否退出
            if input_sentence == 'q' or input_sentence == 'quit': break
            # 句子归一化
            input_sentence = normalizeString(input_sentence)
            # 生成响应Evaluate sentence
            output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
            # 去掉EOS后面的内容
            words = []
            for word in output_words:
                if word == 'EOS':
                    break
                elif word != 'PAD':
                    words.append(word)
            print('Bot:', ' '.join(words))

        except KeyError:
            print("Error: Encountered unknown word.")