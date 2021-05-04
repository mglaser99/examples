###############################################################################
# Language Modeling on Wikitext-2
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import argparse

import torch

import data

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--words', type=int, default='1000',
                    help='number of words to generate')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
# my addition
parser.add_argument('--input', type=str, required=False, default='',
					help='input sequence')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f).to(device)
model.eval()

corpus = data.Corpus(args.data)
ntokens = len(corpus.dictionary)

is_transformer_model = hasattr(model, 'model_type') and model.model_type == 'Transformer'
if not is_transformer_model:
    hidden = model.init_hidden(1)
# input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)

# my additions
# check if there is --input
if args.input == '':
    input_words = []
    input_idxs = None
    input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)
else:
    # split the input into a list and generate the idxs for each input word
    input_words = args.input.split()
    input_idxs = []
    for word in input_words:
        if word not in corpus.dictionary.word2idx:
            raise Exception(f"{word} is not part of the models vocabulary")
        idx = corpus.dictionary.word2idx[word]
        input_idxs.append(torch.tensor([[idx]], dtype=torch.long).to(device))

with open(args.outf, 'w') as outf:

    with torch.no_grad():  # no tracking history

        # my additions
        # add each word of the input to the hidden layers and write them in the file
        if input_idxs is not None:
            for i in range(len(input_idxs)):
                word = corpus.dictionary.idx2word[input_idxs[i]]
                outf.write(word + ('\n' if i % 20 == 19 else ' '))

                output, hidden = model(input_idxs[i], hidden)

            word_weights = output.squeeze().div(args.temperature).exp().cpu()
            word_idx = torch.multinomial(word_weights, 1)[0]
            input = torch.tensor([[word_idx]], dtype=torch.long)

            word = corpus.dictionary.idx2word[word_idx]

            outf.write(word + ('\n' if i % 20 == 19 else ' '))

        for i in range(len(input_words), args.words):
            if is_transformer_model:
                output = model(input, False)
                word_weights = output[-1].squeeze().div(args.temperature).exp().cpu()
                word_idx = torch.multinomial(word_weights, 1)[0]
                word_tensor = torch.Tensor([[word_idx]]).long().to(device)
                input = torch.cat([input, word_tensor], 0)
            else:
                output, hidden = model(input, hidden)
                word_weights = output.squeeze().div(args.temperature).exp().cpu()
                word_idx = torch.multinomial(word_weights, 1)[0]
                input.fill_(word_idx)

            word = corpus.dictionary.idx2word[word_idx]

            outf.write(word + ('\n' if i % 20 == 19 else ' '))

            if i % args.log_interval == 0:
                print('| Generated {}/{} words'.format(i, args.words))
