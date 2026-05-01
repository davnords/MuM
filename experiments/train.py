import sys
from pathlib import Path

import mum.train as training
from mum.utils.trainer import Trainer
from mum.utils.submit import get_args_parser, submit_jobs

def main():
    description = "Submitit launcher for MuM training"
    train_args_parser = training.get_args_parser()
    
    parents = [train_args_parser]
    args_parser = get_args_parser(description=description, parents=parents, add_help=False)
    args = args_parser.parse_args()
    args.training_module = training.__name__

    args.output_dir = Path('output_dir/mum') / args.name / "%j"
    submit_jobs(Trainer, args, name='mum:train')

if __name__ == "__main__":
    sys.exit(main())
