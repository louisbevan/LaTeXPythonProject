import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from positional_encodings.torch_encodings import PositionalEncoding2D

class Attention(nn.Module):
    def __init__(self, hidden_dim = 512):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        # W1*h_{t-1} + W2*e_i transform (e_i memory bank)
        self.attention = nn.Linear(hidden_dim * 2, hidden_dim)
        # \beta^T vector
        self.v = nn.Parameter(torch.rand(hidden_dim))
        # W3[h_t, C_t] transformation
        self.output_proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, hidden, encoder_outputs):
        # hidden: (batch_size, hidden_dim)
        # encoder_outputs: (batch_size, L = H/8 * W/8 = 80, hidden_dim = 512)
        batch_size, seq_len, _ = encoder_outputs.size()

        # repeat hidden state for each pos in encoder outputs
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)

        # calc attn scores
        energy = torch.tanh(self.attention(torch.cat([hidden, encoder_outputs], dim=2)))
        energy = energy.permute(0, 2, 1)

        # calc attn weights (\alpha_{it})
        v = self.v.repeat(batch_size, 1).unsqueeze(1) # (batch_size, 1, hidden_dim)
        attention_weights = F.softmax(torch.bmm(v, energy), dim=2)

        # calc context vector C_t
        context = torch.bmm(attention_weights, encoder_outputs)

        # combine context with hidden state O_t
        output = torch.tanh(self.output_proj(torch.cat([context.squeeze(1), hidden[:, 0, :]], dim=1)))

        return output, attention_weights


class CNNEncoder(nn.Module):
    def __init__(self, input_channels = 1, output_dim = 512):
        super(CNNEncoder, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.Conv2d(256,256, kernel_size=3, padding=1, stride=1),
            nn.MaxPool2d(kernel_size=(1,2), stride=(1,2)) # stride correct?
        )
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1))
        )
        self.conv_block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(512)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model=512, height=4, width=20):
        super(PositionalEncoding, self).__init__()
        self.cnn = CNNEncoder()
        self.pos_encoder = PositionalEncoding2D(d_model)

    def forward(self, x):
        # x shape: (batch_size, 1, 32, 160)
        # CNN output shape: (batch_size, 512, 4, 20)
        x = self.cnn(x)

        # add positional encoding
        x = x + self.pos_encoder

        # for attention layer, need to unfold to L = H' x W' = 4 * 20 = 80
        # i.e. reshape to (batch_size, L, D) = (batch_size, 80, 512)
        batch_size = x.size(0)
        x = x.permute(0,2,3,1) # (batch_size, 512, 4, 20) -> (batch_size, 4, 20, 512)
        x.reshape(batch_size, -1, 512) # (batch_size, 4, 20, 512) -> (batch_size, 80, 512)
        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim = 256, hidden_dim = 512, num_layers = 2):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # token embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # inital states computation
        self.init_h = nn.Linear(hidden_dim, hidden_dim)
        self.init_c = nn.Linear(hidden_dim, hidden_dim)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=embed_dim + hidden_dim, #embedding + prev attn
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )

        # attn mech
        self.attention = Attention(hidden_dim)

        # output projection W4
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    def init_hidden(self, encoder_outputs):
        # calc inital states from encoder outputs mean
        mean_encoder_out = encoder_outputs.mean(dim=1)
        h0 = torch.tanh(self.init_h(mean_encoder_out))
        c0 = torch.tanh(self.init_c(mean_encoder_out))

        # expand for bidirectional lstm
        h0 = h0.unsqueeze(0).repeat(self.lstm.num_layers * 2, 1, 1)
        c0 = c0.unsqueeze(0).repeat(self.lstm.num_layers * 2, 1, 1)

        return (h0, c0)

    def forward(self, encoder_outputs, targets=None, teacher_forcing_ratio = 0.5):
        batch_size = encoder_outputs.size(0)
        max_len = targets.size(1) if targets is not None else self.max_len

        # initialize outputs tensor
        outputs = torch.zeros(batch_size, max_len, self.vocab_size).to(encoder_outputs.device)

        # get initial hidden states
        hidden = self.init_hidden(encoder_outputs)

        # initialize first input as start token
        input_token = torch.zeros(batch_size, 1).long().to(encoder_outputs.device)
        prev_attention = torch.zeros(batch_size, self.hidden_dim).to(encoder_outputs.device)

        for t in range(max_len):
            # embed current input token
            embedded = self.embedding(input_token)

            # concatenate with prev attn output (input feeding)
            lstm_input = torch.cat([embedded.squeeze(1), prev_attention], dim=1).unsqueeze(1)

            # LSTM forward pass
            lstm_out, hidden = self.lstm(lstm_input, hidden)

            # calc attn
            attention_out, attention_weights = self.attention(lstm_out.squeeze(1), encoder_outputs)
            prev_attention = attention_out

            # calc output probs
            output = self.output_proj(attention_out)
            outputs[:, t] = output

            # teacher forcing: use real target as next input
            if targets is not None and torch.rand(1) < teacher_forcing_ratio:
                input_token = targets[:, t].unsqueeze(1)
            else:
                input_token = output.argmax(1).unsqueeze(1)

            return outputs


class ImageToLatex(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512):
        super(ImageToLatex, self).__init__()

        self.encoder = CNNEncoder()
        self.positional_encoding = PositionalEncoding()
        self.decoder = Decoder(vocab_size, embed_dim, hidden_dim)

    def forward(self, images, targets=None, teacher_forcing_ratio=0.5):
        # encode imagtes
        encoded = self.encoder(images)

        # add pos enc
        memory_bank = self.positional_encoding(encoded)

        # decode
        outputs = self.decoder(memory_bank, targets, teacher_forcing_ratio)

        return outputs


