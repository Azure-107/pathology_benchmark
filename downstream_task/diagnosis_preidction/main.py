import os
import time
import wandb
import pandas as pd

from datasets.Subtyping import Dataset_Subtyping
from utils.options import parse_args
from utils.util import set_seed, CV_Meter
from utils.loss import define_loss
from utils.optimizer import define_optimizer
from utils.scheduler import define_scheduler

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, SequentialSampler


def main(args):
    # set random seed for reproduction
    set_seed(args.seed)
    # create results directory
    if args.evaluate:
        results_dir = args.resume
        test_metrics_list = []
    else:
        results_dir = "/mnt/pool/ovariancancer/mSTAR_results/results_{seed}/{study}/[{model}]/[{feature}]-[{time}]".format(
            seed=args.seed,
            study=args.study,
            model=args.model,
            feature=args.feature,
            time=time.strftime("%Y-%m-%d]-[%H-%M-%S"),
        )
    print("[log dir] results directory: ", results_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    # define dataset
    dataset = Dataset_Subtyping(root=args.root, csv_file=args.csv_file, feature=args.feature, encoder_pipeline=args.encoder_pipeline)
    # training and evaluation
    meter = CV_Meter(dataset.num_folds)
    args.num_classes = dataset.num_classes
    args.n_features = dataset.n_features
    args.num_folds = dataset.num_folds
    print('[dataset] number of folds: ', args.num_folds)
    print("[dataset] feature dimension: ", args.n_features)
    for fold in range(dataset.num_folds):
        if args.evaluate and fold != int(args.resume.split("/")[-2].split("_")[-1]):
            print("[model] skip fold {} for evaluation".format(fold))
            continue
        # init wand logger for this fold
        wandb.init(
            project="HKSTU_pathology",   
            name=f"{args.study}-fold{fold}-[{time.strftime('%Y-%m-%d-%H-%M')}]",
            config={
                "encoder": args.feature,
                "lr": args.lr,
                "epochs": args.num_epoch,
                "weight_decay": args.weight_decay,
                "scheduler": args.scheduler,
                "optimizer": args.optimizer,
            }
        )
        splits = dataset.get_fold(fold) # train, test
        class_weights = dataset.get_class_weights(splits[0]) # class weight tensor, for weigted CE loss  # get class weights for the training set
        args.class_weights = class_weights
        # train_loader = DataLoader(dataset, batch_size=1, num_workers=4, pin_memory=True, sampler=SubsetRandomSampler(splits[0]))
        # val_loader = DataLoader(dataset, batch_size=1, num_workers=4, pin_memory=True, sampler=SequentialSampler(splits[1]))
        # loaders = [train_loader, val_loader]
        loaders = [DataLoader(dataset, batch_size=1, num_workers=4, pin_memory=True, sampler=SubsetRandomSampler(split)) for split in splits]
        #loaders = [DataLoader(dataset, batch_size=1, num_workers=4, pin_memory=True, sampler=SequentialSampler(split)) for split in splits]
        # build model, criterion, optimizer, schedular
        #################################################
        if args.model == "ABMIL":
            from models.ABMIL.network import DAttention
            from models.ABMIL.engine import Engine

            #model = DAttention(n_classes=args.num_classes, dropout=0.25, act="relu", n_features=args.n_features)
            model = DAttention(n_classes=args.num_classes, dropout=False, act="relu", n_features=args.n_features)
            engine = Engine(args, results_dir, fold)
        elif args.model == "TransMIL":
            from models.TransMIL.network import TransMIL
            from models.TransMIL.engine import Engine

            model = TransMIL(n_classes=args.num_classes, dropout=0.25, act="relu", n_features=args.n_features)
            engine = Engine(args, results_dir, fold)
        elif "TransMIL_Pre" in args.model:
            from models.TransMIL_Pre.network import TransMIL
            from models.TransMIL_Pre.engine import Engine

            assert os.path.exists(args.aggregator), "aggregator checkpoint not found at {}".format(args.aggregator)
            checkpoint = torch.load(args.aggregator, map_location="cpu")

            model = TransMIL(n_classes=args.num_classes, dropout=0.25, act="relu", n_features=args.n_features)
            vision_encoder_params = {k.replace("module.", "").replace("vision_encoder.", ""): v for k, v in checkpoint["model_state_dict"].items() if "vision_encoder" in k}
            # Load the parameters into the model, with strict=False to allow for missing or unexpected keys
            load_result = model.load_state_dict(vision_encoder_params, strict=False)
            # Check if there are any missing or unexpected keys
            if load_result.missing_keys:
                print("Warning: Missing keys detected during the loading of vision encoder parameters:", load_result.missing_keys)
            if load_result.unexpected_keys:
                print("Warning: Unexpected keys detected during the loading of vision encoder parameters:", load_result.unexpected_keys)

            # If there are neither missing nor unexpected keys, print a success message
            if not load_result.missing_keys and not load_result.unexpected_keys:
                print("Success: Vision encoder parameters loaded correctly into the model.")
            engine = Engine(args, results_dir, fold)
        else:
            raise NotImplementedError("model [{}] is not implemented".format(args.model))
        print("[model] trained model: ", args.model)
        criterion = define_loss(args)
        print("[model] loss function: ", args.loss) # default: ce
        optimizer = define_optimizer(args, model)
        print("[model] optimizer: ", args.optimizer, args.lr, args.weight_decay)    # default: Adam, 2e-4, 1e-5
        scheduler = define_scheduler(args, optimizer)   # default: cosine
        print("[model] scheduler: ", args.scheduler)
        # start training
        if not args.evaluate:
            if args.num_folds > 1:
                val_scores, best_epoch = engine.learning(model, loaders, criterion, optimizer, scheduler)
                meter.updata(best_epoch, val_scores)
            else:
                val_scores, test_scores, best_epoch = engine.learning(model, loaders, criterion, optimizer, scheduler)
                meter.updata(best_epoch, val_scores, test_scores)
        else:
            if fold == int(args.resume.split("/")[-2].split("_")[-1]):
                print('[model] testing on fold: ', fold)
                test_scores = engine.learning(model, loaders, criterion, optimizer, scheduler)
                print("*****************************")
                print("[test scores] ", test_scores)
                print("*****************************")
                test_scores_cpu = {k: v.item() if hasattr(v, "item") else v for k, v in test_scores.items()}
                test_scores_cpu["fold"] = fold
                test_metrics_list.append(test_scores_cpu)
                break

        wandb.finish()
    if not args.evaluate:
        meter.save(os.path.join(results_dir, "result.csv"))
    if args.evaluate and len(test_metrics_list) > 0:
        save_dir = "/mnt/pool/ovariancancer/mSTAR_results/results_{seed}/{study}/[{model}]/[{feature}]-[{time}]".format(
            seed=args.seed,
            study=args.study,
            model=args.model,
            feature=args.feature,
            time=time.strftime("%Y-%m-%d]-[%H-%M-%S"),
        )
        os.makedirs(save_dir, exist_ok=True)
        df = pd.DataFrame(test_metrics_list)
        df = df.sort_values("fold").drop(columns=["fold"]).T  # transpose: metrics x folds
        df.columns = [f"fold_{i}" for i in range(df.shape[1])]
        df = df.round(4)
        df.to_csv(os.path.join(save_dir, f"test_scores_fold_{fold}.csv"), index=True)


if __name__ == "__main__":

    args = parse_args()
    results = main(args)
    print("finished!")
