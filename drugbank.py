import argparse
from train_val import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--drugbank_root', default=None, type=str, required=True,
                        help='..')
    parser.add_argument('--drugbank_path', default=None, type=str, required=True,
                        help='..')
    parser.add_argument('--batch_size', default=16, type=int, required=True,
                        help='..')
    parser.add_argument('--epochs', default=2, type=int, required=True,
                        help='..')
    parser.add_argument('--lr', default=2e-5, type=float, required=True,
                        help='..')
    parser.add_argument('--weight_decay', default=1e-2, type=float, required=True,
                        help='..')
    parser.add_argument('--num_class', default=2, type=int, required=True,
                        help='..')
    parser.add_argument('--cutoff', default=10.0, type=float, required=False,
                        help='..')
    parser.add_argument('--num_layers', default=6, type=int, required=False,
                        help='..')
    parser.add_argument('--hidden_channels', default=128, type=int, required=False,
                        help='..')
    parser.add_argument('--num_filters', default=128, type=int, required=False,
                        help='..')
    parser.add_argument('--num_gaussians', default=50, type=int, required=False,
                        help='..')
    parser.add_argument('--g_out_channels', default=2, type=int, required=False,
                        help='..')

    parser.add_argument('--load_model_path', default=None, type=str, required=False,
                        help='..')
    parser.add_argument('--save_log_path', default="default", type=str, required=False,
                        help='..')
    args = parser.parse_args()
    model = myModel_graph_sch_cnn(num_class=args.num_class,
                                    cutoff=args.cutoff,
                                    num_layers=args.num_layers, hidden_channels=args.hidden_channels,
                                    num_filters=args.num_filters, num_gaussians=args.num_gaussians,
                                    g_out_channels=args.g_out_channels)
    if args.load_model_path is not None:
        save_model = torch.load(args.load_model_path)
        model_dict = model.state_dict()
        state_dict = {k:v for k,v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    dataset = drugbankDataset(root=args.drugbank_root, path=args.drugbank_path)
    split_idx = dataset.get_idx_split(len(dataset.data.y), int(len(dataset.data.y)*0.8), seed=2)
    train_dataset, valid_dataset =dataset[split_idx['train']], dataset[split_idx['valid']]
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, follow_batch=['pos1', 'pos2'])
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, follow_batch=['pos1', 'pos2'])
    train_eval_drugbank(model, optimizer, train_loader, val_loader, args.epochs , log_path = args.save_log_path)
if __name__ == '__main__':
    main()

