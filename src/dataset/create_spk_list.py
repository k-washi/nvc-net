from pathlib import Path
def create_spk_index_list(input_dir, output_path):
    input_dir = Path(input_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    spk_list = sorted(list(input_dir.glob("*")))
    with open(str(output_path), "w") as f:
        for spk in spk_list:
            f.write(str(spk.stem) + "\n")
    
    
if __name__ == "__main__":
    # python ./src/dataset/create_spk_list.py -i /data/karanovc -o ./results/karanovc_spk_list.txt
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=str, required=True)
    parser.add_argument("-o", "--output_path", type=str, required=True)
    
    args = parser.parse_args()
    create_spk_index_list(args.input_dir, args.output_path)