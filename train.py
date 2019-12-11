import time
import sys
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import Encoder, ModifiedDecoderWithAttention
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import argparse
from transformers import WEIGHTS_NAME, BertConfig,BertModel, BertTokenizer, AdamW


cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead


def main():
    """
    Training and validation.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", default='data/', type=str, required=True,
                        help="folder with data files saved by create_input_files.py")
    parser.add_argument("--output_dir", default='saved_models/', type=str, required=True,
                        help="path to save checkpoints")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--data_name", default='coco_5_cap_per_img_5_min_word_freq', type=str,
                        help="base name shared by data files")
    parser.add_argument("--checkpoint", default=None, type=str,
                        help="path to checkpoint")
    parser.add_argument("--emb_dim", default=512, type=int,
                        help="dimension of word embeddings")
    parser.add_argument("--attention_dim", default=512, type=int,
                        help="dimension of attention linear layers")
    parser.add_argument("--decoder_dim", default=512, type=int,
                        help="dimension of decoder RNN")
    parser.add_argument("--dropout", default=0.5, type=float,
                        help="dimension of word embeddings")
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--epochs", default=120, type=int,
                        help="number of epochs to train for (if early stopping is not triggered)")
    parser.add_argument("--batch_size", default=32, type=int,
                        help="batch size for training and testing")
    parser.add_argument("--workers", default=8, type=int,
                        help="num of workers for data-loading")
    parser.add_argument("--bert_lr", default=5e-5, type=float)
    parser.add_argument("--encoder_lr", default=1e-4, type=float)
    parser.add_argument("--decoder_lr", default=5e-4, type=float)
    parser.add_argument("--grad_clip", default=5, type=float,
                        help="clip gradients at an absolute value of")
    parser.add_argument("--alpha_c", default=1, type=int,
                        help="regularization parameter for 'doubly stochastic attention', as in the paper")
    parser.add_argument("--print_freq", default=100, type=int,
                        help="print training/validation stats every __ batches")
    parser.add_argument("--fine_tune_encoder", action='store_true',
                        help="Whether to finetune the encoder")
    parser.add_argument("--bert_dim", default=768, type=int,
                        help="the hidden size of bert model, 768 for bert base, 1024 for bert large")
    parser.add_argument("--add_bert_to_input", action='store_true',
                        help="Whether to add bert hidden to the input of decoder lstm")
    parser.add_argument("--add_bert_to_init_hidden", action='store_true',
                        help="Whether to use bert hidden to initialize the h0 and c0 of lstm")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    best_bleu4 = 0
    epochs_since_improvement = 0

    # Read word map
    word_map_file = os.path.join(args.data_folder, 'WORDMAP_' + args.data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    config = BertConfig.from_pretrained(args.model_name_or_path)
    bert_model = BertModel.from_pretrained(args.model_name_or_path,
                                        from_tf=bool('.ckpt' in args.model_name_or_path),
                                        config=config)


    # Initialize / load checkpoint
    if args.checkpoint is None:
        decoder = ModifiedDecoderWithAttention(attention_dim=args.attention_dim,
                                       embed_dim=args.emb_dim,
                                       bert_dim=args.bert_dim,
                                       decoder_dim=args.decoder_dim,
                                       vocab_size=len(word_map),
                                       dropout=args.dropout,
                                       add_bert_to_input=args.add_bert_to_input,
                                       add_bert_to_init_hidden=args.add_bert_to_init_hidden)
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=args.decoder_lr)
        encoder = Encoder()
        encoder.fine_tune(args.fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=args.encoder_lr) if args.fine_tune_encoder else None

    else:
        checkpoint = torch.load(args.checkpoint)
        args.start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        if args.fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(args.fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=args.encoder_lr)

    # Move to GPU, if available
    decoder = decoder.to(args.device)
    encoder = encoder.to(args.device)
    bert_model = bert_model.to(args.device)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in bert_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    bert_optimizer = AdamW(optimizer_grouped_parameters, lr=args.bert_lr, eps=1e-8)

    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=0).to(args.device)

    # Custom dataloaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(args.data_folder, args.data_name, 'TRAIN', transform=transforms.Compose([normalize])),
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    print(f'train dataset length {len(train_loader)}')
    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(args.data_folder, args.data_name, 'VAL', transform=transforms.Compose([normalize])),
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    print(f'val dataset length {len(val_loader)}')

    # Epochs
    for epoch in range(args.start_epoch, args.epochs):

        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if args.fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)

        # One epoch's training
        train(train_loader=train_loader,
              encoder=encoder,
              bert_encoder=bert_model,
              decoder=decoder,
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              bert_optimizer=bert_optimizer,
              epoch=epoch,
              args=args)

        # One epoch's validation
        recent_bleu4 = validate(val_loader=val_loader,
                                encoder=encoder,
                                bert_encoder=bert_model,
                                decoder=decoder,
                                criterion=criterion,
                                word_map=word_map,
                                args=args)

        # Check if there was an improvement
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        if is_best:
            save_checkpoint(args.data_name, args.output_dir,epoch, epochs_since_improvement, encoder, decoder, bert_model, encoder_optimizer,
                            decoder_optimizer, bert_optimizer, recent_bleu4)


def train(train_loader, encoder, bert_encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, bert_optimizer, epoch, args):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    decoder.train()  # train mode (dropout and batchnorm is used)
    encoder.train()
    bert_encoder.train()

    decoder.zero_grad()
    encoder.zero_grad()
    bert_encoder.zero_grad()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()

    # Batches
    for i, (imgs, caps, caplens, caps_bert, caplens_bert) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to GPU, if available
        imgs = imgs.to(args.device)
        caps = caps.to(args.device)
        caplens = caplens.to(args.device)
        caps_bert = caps_bert.to(args.device)

        #generate attention mask for bert input
        caplens_bert = caplens_bert.squeeze(-1).to(args.device)
        attention_mask = (torch.arange(len(caps_bert[0]))[None, :] < caplens_bert[:, None]).squeeze(1).float().to(args.device)

        # Forward prop.
        bert_output = bert_encoder(caps_bert, attention_mask=attention_mask)
        imgs = encoder(imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs,bert_output, caps, caplens)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]
        scores = scores.contiguous()
        targets = targets.contiguous()

        # Calculate loss
        loss = criterion(scores.view(-1,scores.size(-1)), targets.view(-1))

        # Add doubly stochastic attention regularization
        loss += args.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # Back prop.
        loss.backward()

        # Clip gradients
        if args.grad_clip is not None:
            clip_gradient(decoder_optimizer, args.grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, args.grad_clip)
            if bert_optimizer is not None:
                clip_gradient(bert_optimizer, args.grad_clip)

        # Update weights
        decoder_optimizer.step()
        encoder_optimizer.step()
        bert_optimizer.step()

        decoder_optimizer.zero_grad()
        encoder_optimizer.zero_grad()
        bert_optimizer.zero_grad()

        # Keep track of metrics
        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))
        sys.stdout.flush()


def validate(val_loader, encoder, bert_encoder, decoder, criterion, word_map,args):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param bert_encoder: encoder model with bert surpervising
    :param decoder: decoder model
    :param criterion: loss layer
    :return: BLEU-4 score
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()
    if bert_encoder is not None:
        bert_encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    # explicitly disable gradient calculation to avoid CUDA memory error
    # solves the issue #57
    with torch.no_grad():
        # Batches
        for i, (imgs, caps, caplens, caps_bert, caplens_bert, allcaps) in enumerate(val_loader):

            # Move to device, if available
            imgs = imgs.to(args.device)
            caps = caps.to(args.device)
            caplens = caplens.to(args.device)
            caps_bert = caps_bert.to(args.device)

            # generate attention mask for bert input
            caplens_bert = caplens_bert.squeeze(-1).to(args.device)
            attention_mask = (torch.arange(len(caps_bert[0]))[None, :] < caplens_bert[:, None]).squeeze(1).float().to(
                args.device)

            # Forward prop.
            bert_output = bert_encoder(caps_bert, attention_mask=attention_mask)
            imgs = encoder(imgs)

            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs,bert_output, caps, caplens)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]
            scores = scores.contiguous()
            targets = targets.contiguous()

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_copy = scores.clone()

            # Calculate loss
            loss = criterion(scores.view(-1,scores.size(-1)), targets.view(-1))

            # Add doubly stochastic attention regularization
            loss += args.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % args.print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                                loss=losses, top5=top5accs))

            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References
            allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        img_caps))  # remove <start> and pads
                references.append(img_captions)

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

        # Calculate scores using different metric
        bleu1, bleu2, bleu3, bleu4, Meteor = get_metrics_scores(references,hypotheses)

        print(
            '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-1 - {bleu1}, BLEU-2 - {bleu2}, BLEU-3 - {bleu3}, BLEU-4 - {bleu4}, Meteor - {Meteor}\n'.format(
                loss=losses,
                top5=top5accs,
                bleu1=bleu1,
                bleu2=bleu2,
                bleu3=bleu3,
                bleu4=bleu4,
                Meteor=Meteor))

    return bleu4


if __name__ == '__main__':
    main()
