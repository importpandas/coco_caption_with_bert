import os
import numpy as np
import h5py
import json
import torch
from scipy.misc import imread, imresize
from tqdm import tqdm
from collections import Counter
import random
from random import seed, choice, sample
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import  meteor_score


def create_input_files(dataset, karpathy_json_path, image_folder, captions_per_image, min_word_freq, output_folder,
                       tokenizer=None,max_len=100):
    """
    Creates input files for training, validation, and test data.

    :param dataset: name of dataset, one of 'coco', 'flickr8k', 'flickr30k'
    :param karpathy_json_path: path of Karpathy JSON file with splits and captions
    :param image_folder: folder with downloaded images
    :param captions_per_image: number of captions to sample per image
    :param min_word_freq: words occuring less frequently than this threshold are binned as <unk>s
    :param output_folder: folder to save files
    :param max_len: don't sample captions longer than this length
    """

    assert dataset in {'coco', 'flickr8k', 'flickr30k'}

    # Read Karpathy JSON
    with open(karpathy_json_path, 'r') as j:
        data = json.load(j)

    # Read image paths and captions for each image
    train_image_paths = []
    train_image_captions = []
    train_image_captions_bert = []
    val_image_paths = []
    val_image_captions = []
    val_image_captions_bert = []
    test_image_paths = []
    test_image_captions = []
    test_image_captions_bert = []
    word_freq = Counter()

    for img in data['images']:
        captions = []
        bert_captions = []
        for c in img['sentences']:
            # Update word frequency
            word_freq.update(c['tokens'])
            if len(c['tokens']) <= max_len:
                captions.append(c['tokens'])
                if tokenizer is not None:
                    bert_caption_tokens = tokenizer.tokenize(c['raw'])
                    bert_captions.append(bert_caption_tokens)


        if len(captions) == 0:
            continue

        path = os.path.join(image_folder, img['filepath'], img['filename']) if dataset == 'coco' else os.path.join(
            image_folder, img['filename'])

        if img['split'] in {'train', 'restval'}:
            train_image_paths.append(path)
            train_image_captions.append(captions)
            if tokenizer is not None:
                train_image_captions_bert.append(bert_captions)
        elif img['split'] in {'val'}:
            val_image_paths.append(path)
            val_image_captions.append(captions)
            if tokenizer is not None:
                val_image_captions_bert.append(bert_captions)
        elif img['split'] in {'test'}:
            test_image_paths.append(path)
            test_image_captions.append(captions)
            if tokenizer is not None:
                test_image_captions_bert.append(bert_captions)

    # Sanity check
    assert len(train_image_paths) == len(train_image_captions)
    assert len(val_image_paths) == len(val_image_captions)
    assert len(test_image_paths) == len(test_image_captions)

    # Create word map
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    # Create a base/root name for all output files
    base_filename = dataset + '_' + str(captions_per_image) + '_cap_per_img_' + str(min_word_freq) + '_min_word_freq'

    # Save word map to a JSON
    with open(os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json'), 'w') as j:
        json.dump(word_map, j)

    # Sample captions for each image, save images to HDF5 file, and captions and their lengths to JSON files
    seed(123)
    for impaths, imcaps, imcaps_bert, split in [(train_image_paths, train_image_captions, train_image_captions_bert, 'TRAIN'),
                                   (val_image_paths, val_image_captions, val_image_captions_bert, 'VAL'),
                                   (test_image_paths, test_image_captions, test_image_captions_bert, 'TEST')]:

        with h5py.File(os.path.join(output_folder, split + '_IMAGES_' + base_filename + '.hdf5'), 'a') as h:
            # Make a note of the number of captions we are sampling per image
            h.attrs['captions_per_image'] = captions_per_image

            # Create dataset inside HDF5 file to store images
            images = h.create_dataset('images', (len(impaths), 3, 256, 256), dtype='uint8')

            print("\nReading %s images and captions, storing to file...\n" % split)

            enc_captions = []
            caplens = []

            enc_captions_bert = []
            caplens_bert = []

            for i, path in enumerate(tqdm(impaths)):

                # Sample captions
                if len(imcaps[i]) < captions_per_image:
                    random_choices = [random.randrange(0,len(imcaps[i])) for _ in range(captions_per_image - len(imcaps[i]))]
                    captions = imcaps[i] + [imcaps[i][j]  for j in random_choices]
                    captions_bert = imcaps_bert[i] + [imcaps_bert[i][j] for j in random_choices]
                else:
                    random_samples = random.sample(list(range(len(imcaps[i]))), k=captions_per_image)
                    captions = [imcaps[i][j] for j in random_samples]
                    captions_bert = [imcaps_bert[i][j] for j in random_samples]

                # Sanity check
                assert len(captions) == captions_per_image
                assert len(captions_bert) == captions_per_image

                # Read images
                img = imread(impaths[i])
                if len(img.shape) == 2:
                    img = img[:, :, np.newaxis]
                    img = np.concatenate([img, img, img], axis=2)
                img = imresize(img, (256, 256))
                img = img.transpose(2, 0, 1)
                assert img.shape == (3, 256, 256)
                assert np.max(img) <= 255

                # Save image to HDF5 file
                images[i] = img

                for j, c in enumerate(captions):
                    # Encode captions
                    enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                        word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))

                    # Find caption lengths
                    c_len = len(c) + 2

                    enc_captions.append(enc_c)
                    caplens.append(c_len)

                if tokenizer is not None:
                    for j, c in enumerate(captions_bert):
                        if len(c) > max_len:
                            c = c[0:max_len]
                        tokens = []
                        tokens.append("[CLS]")
                        tokens += c
                        tokens.append("[SEP]")
                        enc_c_bert = tokenizer.convert_tokens_to_ids(tokens) + (max_len - len(c)) * [0]

                        enc_captions_bert.append(enc_c_bert)
                        caplens_bert.append(len(tokens))

            # Sanity check
            assert images.shape[0] * captions_per_image == len(enc_captions) == len(caplens)
            assert images.shape[0] * captions_per_image == len(enc_captions_bert) == len(caplens_bert)

            # Save encoded captions and their lengths to JSON files
            with open(os.path.join(output_folder, split + '_CAPTIONS_' + base_filename + '.json'), 'w') as j:
                json.dump(enc_captions, j)

            with open(os.path.join(output_folder, split + '_CAPLENS_' + base_filename + '.json'), 'w') as j:
                json.dump(caplens, j)

            if tokenizer is not None:
                with open(os.path.join(output_folder, split + '_CAPTIONS_BERT_' + base_filename + '.json'), 'w') as j:
                    json.dump(enc_captions_bert, j)

                with open(os.path.join(output_folder, split + '_CAPLENS_BERT_' + base_filename + '.json'), 'w') as j:
                    json.dump(caplens_bert, j)


def init_embedding(embeddings):
    """
    Fills embedding tensor with values from the uniform distribution.

    :param embeddings: embedding tensor
    """
    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)


def load_embeddings(emb_file, word_map):
    """
    Creates an embedding tensor for the specified word map, for loading into the model.

    :param emb_file: file containing embeddings (stored in GloVe format)
    :param word_map: word map
    :return: embeddings in the same order as the words in the word map, dimension of embeddings
    """

    # Find embedding dimension
    with open(emb_file, 'r') as f:
        emb_dim = len(f.readline().split(' ')) - 1

    vocab = set(word_map.keys())

    # Create tensor to hold embeddings, initialize
    embeddings = torch.FloatTensor(len(vocab), emb_dim)
    init_embedding(embeddings)

    # Read embedding file
    print("\nLoading embeddings...")
    for line in open(emb_file, 'r'):
        line = line.split(' ')

        emb_word = line[0]
        embedding = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))

        # Ignore word if not in train_vocab
        if emb_word not in vocab:
            continue

        embeddings[word_map[emb_word]] = torch.FloatTensor(embedding)

    return embeddings, emb_dim


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(data_name, output_dir,epoch, epochs_since_improvement, encoder, decoder, bert_encoder, encoder_optimizer, decoder_optimizer,
                    bert_optimizer, bleu4):
    """
    Saves model checkpoint.

    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    :param bleu4: validation BLEU-4 score for this epoch
    :param is_best: is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'bleu-4': bleu4,
             'encoder': encoder,
             'decoder': decoder,
             'bert_encoder':bert_encoder,
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer,
             'bert_optimizer': bert_optimizer}
    filename = 'checkpoint_'+ str(bleu4) + '_'+ data_name + '.pth.tar'
    saved_path = os.path.join(output_dir, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    torch.save(state, saved_path)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    _, ind = scores.topk(k, 2, True, True)
    ind = ind.view(-1,ind.size(-1))
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / (targets.size(0)*targets.size(1)) )

def intlist2strlist(intlist):
    return [str(val) for val in intlist]

def corpus_meteor(list_of_refs,list_of_hypos):
    # the original input format of Meteor metric is different form BLEU series
    # in this function, we change the format of BLEU to fit Meteor
    Meteor = 0.0

    for i,ref in enumerate(list_of_refs):
        ref_list_tmp = [' '.join(intlist2strlist(val)) for val in ref]
        hypo_tmp = ' '.join(intlist2strlist(list_of_hypos[i]))
        Meteor += meteor_score(ref_list_tmp,hypo_tmp)

    return Meteor / (len(list_of_hypos))


def get_metrics_scores(references, hypotheses):
    '''
    :param references: references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...]
    :param hypotheses: hypotheses = [hyp1, hyp2, ...]
    :return: the scores using five different metrics
    '''

    bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))
    bleu3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0))
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))

    Meteor = corpus_meteor(references, hypotheses)

    return (bleu1, bleu2, bleu3, bleu4, Meteor)
