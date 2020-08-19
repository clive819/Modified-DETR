import os
from argparse import ArgumentParser

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.dataloader import DataLoader

from models import DETR, SetCriterion
from utils.dataset import YOLODataset, collateFunction
from utils.misc import baseParser, MetricsLogger, saveArguments, logMetrics


def main(args):
    print(args)
    saveArguments(args, args.taskName)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    if not os.path.exists(args.outputDir):
        os.mkdir(args.outputDir)

    # load data
    dataset = YOLODataset(args.dataDir, args.targetHeight, args.targetWidth, args.numClass)
    dataLoader = DataLoader(dataset, batch_size=args.batchSize, shuffle=True, collate_fn=collateFunction,
                            pin_memory=True, num_workers=args.numWorkers)

    # load model
    model = DETR(args).to(device)
    criterion = SetCriterion(args).to(device)

    # resume training
    if args.weight and os.path.exists(args.weight):
        print(f'loading pre-trained weights from {args.weight}')
        model.load_state_dict(torch.load(args.weight, map_location=device))

    # multi-GPU training
    model = torch.nn.DataParallel(model)

    # separate learning rate
    paramDicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lrBackbone,
        },
    ]

    optimizer = AdamW(paramDicts, args.lr, weight_decay=args.weightDecay)
    lrScheduler = StepLR(optimizer, args.lrDrop)
    prevBestLoss = np.inf
    batches = len(dataLoader)
    logger = MetricsLogger()

    model.train()
    criterion.train()

    for epoch in range(args.epochs):
        losses = []
        for batch, (x, y) in enumerate(dataLoader):
            x = x.to(device)
            y = [{k: v.to(device) for k, v in t.items()} for t in y]

            out = model(x)
            metrics = criterion(out, y)

            loss = sum(v for k, v in metrics.items() if 'loss' in k)
            losses.append(loss.cpu().item())

            # MARK: - print & save training details
            print(f'Epoch {epoch} | {batch + 1} / {batches}')
            logMetrics(metrics)
            logger.step(metrics, epoch, batch)

            # MARK: - backpropagation
            optimizer.zero_grad()
            loss.backward()
            if args.clipMaxNorm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clipMaxNorm)
            optimizer.step()

        lrScheduler.step()
        logger.epochEnd(epoch)
        avgLoss = np.mean(losses)
        print(f'Epoch {epoch}, loss: {avgLoss:.8f}')

        if avgLoss < prevBestLoss:
            print('[+] Loss improved from {:.8f} to {:.8f}, saving model...'.format(prevBestLoss, avgLoss))
            if not os.path.exists(args.outputDir):
                os.mkdir(args.outputDir)

            try:
                stateDict = model.module.state_dict()
            except AttributeError:
                stateDict = model.state_dict()
            torch.save(stateDict, f'{args.outputDir}/{args.taskName}.pt')
            prevBestLoss = avgLoss
            logger.addScalar('Model', avgLoss, epoch)
        logger.flush()
    logger.close()


if __name__ == '__main__':
    parser = ArgumentParser('python3 train.py', parents=[baseParser()])

    # MARK: - training config
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lrBackbone', default=1e-4, type=float)
    parser.add_argument('--batchSize', default=32, type=int)
    parser.add_argument('--weightDecay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=30000, type=int)
    parser.add_argument('--lrDrop', default=20000, type=int)
    parser.add_argument('--clipMaxNorm', default=.1, type=float)

    # MARK: - loss
    parser.add_argument('--classCost', default=1., type=float)
    parser.add_argument('--bboxCost', default=5., type=float)
    parser.add_argument('--giouCost', default=2., type=float)
    parser.add_argument('--eosCost', default=.1, type=float)

    # MARK: - dataset
    parser.add_argument('--dataDir', default='/home/clive819/toy', type=str)

    # MARK: - miscellaneous
    parser.add_argument('--outputDir', default='./checkpoint', type=str)
    parser.add_argument('--taskName', default='toy', type=str)
    parser.add_argument('--numWorkers', default=16, type=int)

    main(parser.parse_args())
