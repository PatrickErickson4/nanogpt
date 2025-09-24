import torch
import torch.nn as nn
from torch.nn import functional as F

with open('simple_wiki.txt', 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(list(set(text)))

### HYPERPARAMS
device = 'cuda' if torch.cuda.is_available() else 'cpu'
contextLength = 256 # TODO: GLOBAL VAR (HYPERPARAM)
batchSize = 64 # TODO: HYPERPARM 2
epochs = 5000
lossInterval = 500
lr = 3e-4
generateMaxTokens = 200
nEmbed = 384
nHeads = 6
nLayer = 6
dropout = .2

torch.manual_seed(100)

# sets up character,index pair dictionary for 
# character encoder
oneHotEncoder = {character:index for index,character in enumerate(chars)}

# constructs the character decoder for dictionaries
oneHotDecoder = {index:character for index,character in enumerate(chars)}
encode = lambda x: [oneHotEncoder[i] for i in x]
decode = lambda y: ''.join([oneHotDecoder[i] for i in y])


vocabSize = len(oneHotDecoder) # HYPERPARAM

# split dataset
df = torch.tensor(encode(text),dtype=torch.long)
trainSize = int(.95*len(df))
train = df[:trainSize]
test = df[trainSize:]


# chosen to size 8 arbitrarily. Think of this as a convolving filter. 
# we do contextLength + 1 in order to account for the first item
# in the context. We can't predict a character if we have nothing to 
# predict off of!
# NOTE: We sometimes overlap context lengths over epochs to ensure that we 
# don't always divide at some modulo n. 


# print(train[:contextLength+1])

'''
x = train[:contextLength+1]
y = train[1:contextLength+1] # see?

# NOTE: t stands for the location in the TIME dimension

for t in range(contextLength):
    context = x[:t+1]
    targetPrediction = y[t]
    print(f"When input is {context} the expected output should be {targetPrediction}")
'''
@torch.no_grad()
def estimateLoss():
    toRet = {}
    LLM.eval()
    for split in ['train','test']:
        losses = torch.zeros(generateMaxTokens)
        for i in range(generateMaxTokens):
            x, y = getBatch(split)
            _,losses[i] = LLM(x, y)
        toRet[split] = losses.mean()
    LLM.train()
    return toRet

def getBatch(split):

    # We want to get the batch of the relevant data
    data = train if split == 'train' else test
    # picks some tensor (8 batches of 16 context length pieces of data)
    # this is with replacement. This is because the corpus is so large
    # deterministic approaches are not necessarily needed for pretraining
    index = torch.randint(len(data) - contextLength, (batchSize,))
    x = torch.stack([data[i:i+contextLength] for i in index])
    y = torch.stack([data[i+1:i+contextLength+1] for i in index])
    x, y = x.to(device), y.to(device)
    return x, y

'''
print("inputs:\n",xBatch)
print("inputs:\n",yBatch)
'''

class SelfAttention(nn.Module):
    def __init__(self, headSize):
        super().__init__()

        self.key = nn.Linear(nEmbed,headSize,bias=False)
        self.query = nn.Linear(nEmbed,headSize,bias=False)
        self.value = nn.Linear(nEmbed,headSize,bias=False)

        # think of this as a pre-definited "mask" of a triangular 1's and 0's matrix.
        # this ensures that every token cant talk to its future tokens.
        self.register_buffer('tril', torch.tril(torch.ones(contextLength,contextLength)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        _, timeDim, vocabDim = x.shape
        key = self.key(x)
        query = self.query(x)
        value = self.value(x)

        # thhis is qkT/sqrt(d) in attention is all you need
        weights =  query @ key.transpose(-2,-1) * vocabDim**-0.5 # (B, T , C) @ (B, C ,T) ---> (B)
        
        # this turns it into a decoder block
        weights = weights.masked_fill(self.tril[:timeDim,:timeDim] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        toRet = weights @ value
        return toRet

class MultiHeadAttention(nn.Module):
    def __init__(self, numHeads, headSize):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttention(headSize) for _ in range(numHeads)])
        self.projection = nn.Linear(nEmbed,nEmbed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        toRet = torch.cat([h(x) for h in self.heads], dim = -1)
        toRet = self.projection(toRet)
        toRet = self.dropout(toRet)
        return toRet
    
class MLP(nn.Module):
    def __init__(self, nEmbed):
        super().__init__()
        self.mlp = nn.Sequential(
            # adding computation to learn more representations
            # attention is all you need
            nn.Linear(nEmbed,4 * nEmbed),
            nn.ReLU(),
            # projection layer that goes back into the residual pathway
            nn.Linear(4 * nEmbed,nEmbed),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.mlp(x)

class Block(nn.Module):
    def __init__(self, nEmbed, nHeads):
        super().__init__()
        headSize = nEmbed // nHeads
        self.saHeads = MultiHeadAttention(nHeads, headSize)
        self.mlp = MLP(nEmbed)
        self.layerNorm1 = nn.LayerNorm(nEmbed)
        self.layerNorm2 = nn.LayerNorm(nEmbed)


    def forward(self, x):
        # THIS IS DEVIANT FROM ATTENTION IS ALL YOU NEED
        x = x + self.saHeads(self.layerNorm1(x))
        x = x + self.mlp(self.layerNorm2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # nEmbedding dimensional embeds -> We're making 
        # dense vectors like we did in pos tagging
        self.embeddingTable = nn.Embedding(vocabSize,nEmbed)
        self.positionalEncoder = nn.Embedding(contextLength,nEmbed)
        # implement multi-headed attention. We divide nEmbed by however many
        # self-attention layers we want
        self.blocks = nn.Sequential(*[Block(nEmbed, nHeads=nHeads) for _ in range(nLayer)])
        #constructing a linear layer for the dense vector!
        self.layerNormFinal = nn.LayerNorm(nEmbed)
        self.lmHead = nn.Linear(nEmbed, vocabSize) 
    def forward(self, index, targets=None):

        _,timeDim = index.shape
        tokenEmbeddings = self.embeddingTable(index)
        positionalEmbeddings = self.positionalEncoder(torch.arange(timeDim, device=device))
        x = tokenEmbeddings + positionalEmbeddings
        x = self.blocks(x)
        x = self.layerNormFinal(x)
        logits = self.lmHead(x)

        if targets is None:
            loss = None
        else:
            batchDim, timeDim, channelDim = logits.shape
            # reshaping for matmuls in crossEntropy
            logits = logits.view(batchDim*timeDim,channelDim)
            targets = targets.view(batchDim*timeDim)
            loss = F.cross_entropy(logits,targets)
        return logits, loss 
    
    def generate(self, index, maxNewTokens):

        # generate specified number of tokens
        for _ in range(maxNewTokens):
            indexCondition = index[:,-contextLength:]

            logits,_ = self(indexCondition)
            # we're focusing on the last prediction
            logits = logits[:,-1,:]
            # apply softmax to get the probability dist
            probDist = F.softmax(logits, dim=-1)
            # randomly sample from the softmax probability distribution
            prediction = torch.multinomial(probDist, num_samples=1) # (B,1)
            # We concatenate at the end and preserve the dimension (B, T+1)
            index = torch.cat((index, prediction), dim=1) # along time dimension!
        return index # updated generated point!



LLM = GPTLanguageModel()
LLM = LLM.to(device)


index = torch.zeros((1,1),dtype=torch.long,device=device) # our <START> token! its to make sure 
# that we generate the first characters for generation
print("Before training:")
print(decode(LLM.generate(index,maxNewTokens=generateMaxTokens)[0].tolist()))
print("")

optimizer = torch.optim.AdamW(LLM.parameters(), lr = lr)
for steps in range(epochs):

    if steps % lossInterval == 0:
        losses = estimateLoss()
        print(f"At epoch {steps}: training loss: {losses['train']:.4f}, validation loss: {losses['test']:.4f}")

    # grab some randomly generated batch
    xBatch,yBatch = getBatch('train')
    logits,loss = LLM(xBatch,yBatch)
    # this sets all gradients for backprop to 0
    # for faster optim
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
print("Final Validation Loss: ", loss.item())
print("After training:")
print(decode(
    LLM.generate(
        index,
        maxNewTokens=generateMaxTokens)[0].tolist())) 

with open('final.txt', 'w', errors='ignore') as f:
    f.write(decode(LLM.generate(index, maxNewTokens=10000)[0].tolist()))