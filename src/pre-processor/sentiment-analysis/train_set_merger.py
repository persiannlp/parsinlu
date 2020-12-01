import argparse
import os

parser = argparse.ArgumentParser(description='SA TrainSet Merger')
parser.add_argument('--domains', nargs='+', type=str, default=['food', 'movie'],
                        help='Specify the domains you want to merge.')

parser.add_argument('--input_dir', type=str, help='Specify input directory to load the files')
parser.add_argument('--output_dir', type=str, help='Specify output directory to save the files')

args = parser.parse_args()
print(' '.join(['Merging TrainSets from:'] + args.domains))

for dataset_split_name in ['train']:
    with open (os.path.join(args.output_dir, f'ABSA_Dataset_{dataset_split_name}.jsonl') ,'w') as out_file:
        for domain in args.domains:
            with open (os.path.join(args.input_dir, f'{domain}_{dataset_split_name}.jsonl')) as in_file:
                for line in in_file.readlines():
                    out_file.write(line.strip()+'\n')

print("TrainSets are merged successfully.")