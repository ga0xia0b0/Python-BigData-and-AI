import numpy as np
import pandas as pd
import re
from collections import Counter
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
train = pd.read_csv('train.tsv', sep='\t')
test = pd.read_csv('test.tsv', sep='\t')
g = open('train.tsv','r')
train = list(map(lambda x: x[:-1], g.readlines()))
g.close()
g = open('test.tsv','r')
test = list(map(lambda x: x[:-1], g.readlines()))
g.close()
linha_1 = train[1]
lista_teste = linha_1.split('\t')
train_lines = []
for i in range(len(train)):
    train_lines.append(train[i])
train_set_lines = []
for i in range(len(train_lines)):
    dados_linha = train_lines[i]
    dados_linha = dados_linha.split('\t')
    train_set_lines.append(dados_linha)
test_lines = []
for i in range(len(test)):
    test_lines.append(test[i])
test_set_lines = []
for i in range(len(test_lines)):
    dados_linha = test_lines[i]
    dados_linha = dados_linha.split('\t')
    test_set_lines.append(dados_linha)
for i in range(1, len(train_set_lines)):
    train_set_lines[i][3] = int(train_set_lines[i][3])

for i in range(len(train_set_lines)):
    train_set_lines[i][2] = train_set_lines[i][2].lower()
    string = train_set_lines[i][2]
    train_set_lines[i][2] = re.sub('[^a-z0-9\s]', '', string)
for i in range(len(test_set_lines)):
    test_set_lines[i][2] = test_set_lines[i][2].lower()
    string = test_set_lines[i][2]
    test_set_lines[i][2] = re.sub('[^a-z0-9\s]', '', string)

all_text = []
for i in range(1,len(train_set_lines)):
    all_text.append(train_set_lines[i][2])
for i in range(1,len(test_set_lines)):
    all_text.append(test_set_lines[i][2])
all_text = ' '.join(all_text)
words = all_text.split()
counts = Counter(words)
vocabulary = sorted(counts, key=counts.get, reverse=True)
vocabulary_to_int = {}
int_to_vocabulary = {}
for i, word in enumerate(vocabulary, 1):
    vocabulary_to_int[word] = i
    int_to_vocabulary[i] = word

train_set_int = []
for review in train_set_lines:
    if review[0] == 'PhraseId':
        continue
    train_set_int.append([vocabulary_to_int[word] for word in review[2].split()])
encoded_labels = np.array([train_set_lines[i][3] for i in range(1,len(train_set_lines))])

test_set_int = []
for review in test_set_lines:
    if review[0] == 'PhraseId':
        continue
    test_set_int.append([vocabulary_to_int[word] for word in review[2].split()])
review_49 = train_set_int[49] # review em formato token
review_49_words = ' '.join([int_to_vocabulary[i] for i in review_49]) # refazendo a conversão

tamanho_reviews = Counter([len(x) for x in train_set_int]) # Dict: {objeto: contagem}
tamanho_reviews_test = Counter([len(x) for x in test_set_int])
tamanho = sorted([key for key in tamanho_reviews.keys()])
numero = [tamanho_reviews[lenght] for lenght in tamanho]
plt.plot(tamanho, numero)
plt.xlabel('size')
plt.ylabel('quantity')
plt.show()

zero_idx_train = [i for i, review in enumerate(train_set_int) if len(review) == 0]
encoded_labels_null = [encoded_labels[i] for i in zero_idx_train]
count_label_null = Counter(encoded_labels_null)
x_data = [0,1,2,3,4]
y_data = [0,1,26,0,0]
plt.plot(x_data, y_data)
plt.show()

non_zero_idx_train = [i for i, review in enumerate(train_set_int) if len(review) != 0]
non_zero_idx_test = [i for i, review in enumerate(test_set_int) if len(review) != 0]
train_set_int = [train_set_int[i] for i in non_zero_idx_train]
encoded_labels = np.array([encoded_labels[i] for i in non_zero_idx_train])
test_set_int = [test_set_int[i] for i in non_zero_idx_test]
def padding_features(review_ints, seq_length):
    features = np.zeros((len(review_ints), seq_length), dtype=int)
    for i, row in enumerate(review_ints):
        features[i, -len(row):] = np.array(row[:seq_length])
    return features
seq_length = 52
features_train = padding_features(train_set_int, seq_length=seq_length)
features_test = padding_features(test_set_int, seq_length=seq_length)

split_frac = 0.8
split_idx = int(len(features_train)*split_frac)
train_x, remaining_x = features_train[:split_idx], features_train[split_idx:]
train_y, remaining_y = encoded_labels[:split_idx], encoded_labels[split_idx:]
test_idx = int(len(remaining_x)*0.5)
val_x, test_x = remaining_x[:test_idx], remaining_x[test_idx:]
val_y, test_y = remaining_y[:test_idx], remaining_y[test_idx:]

train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
valid_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

batch_size = 128
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size, drop_last=True)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size, drop_last=True)

dataiter = iter(train_loader)
sample_x, sample_y = dataiter.__next__()
train_on_gpu=torch.cuda.is_available()

if(train_on_gpu):
    print('Cheating on GPU.')
else:
    print('GPU not available, training on the CPU.')
class SentimentRNN(nn.Module):
    """
    Modelo de uma rede recorrente usada para realizar analise de sentimentos.
    """

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        # Aqui vamos inicializar a rede e os seus parâmetros

        super(SentimentRNN, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # Camadas de embedding e da LSTM
        # Na LSTM, deve-se colocar batch_first pois os tensores
        # de input e output tem a forma: (batch, seq, feature)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                            dropout=drop_prob, batch_first=True)

        # Dropout
        self.dropout = nn.Dropout(p=0.3)

        # Camada linear, dropout e softmax
        self.fc = nn.Linear(hidden_dim, output_size)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, hidden):
        # Essa função é para realizar o forward pass do modelo no input e nos hidden states

        batch_size = x.size(0)

        # Embeddings e lstm_out
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)

        # Juntando os outputs da LSTM
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        # Dropout e a camada totalmente conectada
        out = self.dropout(lstm_out)
        out = self.fc(out)
        # Função LogSoftmax
        soft_out = self.log_softmax(out)

        # Reshape para o batchfirst
        soft_out = soft_out.view(batch_size, -1, self.output_size)
        soft_out = soft_out[:, -1, :]  # pega as labels(o output) da ultima posiçao te todas as linhas

        # Retorna o último output da softmax e do hidden state
        return soft_out, hidden

    def init_hidden(self, batch_size):
        # Essa função inicializa o hidden state

        # Cria dois novos tensores de tamanho n_layers x batch_size x hidden_dim,
        # inicializados com zeros, pra o hidden state e a cell state da LSTM
        weight = next(self.parameters()).detach()

        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())

        return hidden

vocab_size = len(vocabulary_to_int) + 1 # +1 pelo padding de 0's
output_size = 5
embedding_dim = 200
hidden_dim = 256
n_layers = 2
net = SentimentRNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)

print(net)
lr = 0.001
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
epochs = 4
counter = 0
print_every = 100
clip = 5
if(train_on_gpu):
    net.cuda()
net.train()
for e in range(epochs):
    h = net.init_hidden(batch_size)
    losses = []
    for inputs, labels in train_loader:
        if ((inputs.shape[0], inputs.shape[1]) != (batch_size, seq_length)):
            continue
        counter += 1
        if(train_on_gpu):
           inputs, labels = inputs.cuda(), labels.cuda()
        h = tuple([each.detach() for each in h])
        net.zero_grad()
        output, h = net(inputs, h)
        loss = criterion(output.squeeze(), labels.long())
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()
        if counter % print_every == 0:
            val_h = net.init_hidden(batch_size)
            val_losses = []
            net.eval()
            for inputs, labels in valid_loader:
                val_h = tuple([each.detach() for each in val_h])
                if(train_on_gpu):
                   inputs, labels = inputs.cuda(), labels.cuda()
                output, val_h = net(inputs, val_h)
                val_loss = criterion(output.squeeze(), labels.long())
                val_losses.append(val_loss.item())
            net.train()
            print("Epoch: {}/{}...".format(e + 1, epochs),
                  "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format(loss.item()),
                  "Val Loss: {:.6f}".format(np.mean(val_losses)))
plt.plot(val_losses)
plt.show()
torch.save(net.state_dict(), 'sentiment_net.pt')
net.load_state_dict(torch.load('sentiment_net.pt'))

test_loss = 0.0
class_correct = list(0. for i in range(5))
class_total = list(0. for i in range(5))
h = net.init_hidden(batch_size)
net.eval()
for inputs, labels in test_loader:
    h = tuple([each.data for each in h])
    if(train_on_gpu):
        inputs, labels = inputs.cuda(), labels.cuda()
    output, h = net(inputs, h)
    loss = criterion(output, labels.long())
    test_loss += loss.item() * inputs.size(0)
    _, pred = torch.max(output, 1)
    correct = np.squeeze(pred.eq(labels.data.view_as(pred)))
    for i in range(len(labels)):
        label = labels.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1
test_loss = test_loss / len(test_loader.sampler)
print('Test Loss: {:.6f}\n'.format(test_loss))
for i in range(5):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            str(i), 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (class_total[i]))
print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (100. * np.sum(class_correct) / np.sum(class_total), np.sum(class_correct), np.sum(class_total)))