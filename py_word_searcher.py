import os
import argparse
import pandas as pd


def search_word_in_files(word, root_dir, file_type):
    detections = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(file_type):
                file_path = os.path.join(subdir, file)
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line_number, line in enumerate(f, 1):
                        if word in line:
                            detections.append([file_path, line_number, line])

    return detections

def main():
    parser = argparse.ArgumentParser(description='Look for location')
    parser.add_argument('-bd', '--base_root_directory', type=str, default='')
    parser.add_argument('-wd', '--search_word', type=str, default='')
    parser.add_argument('-ft', '--file_type', type=str, default='.py', help='File type: . + extension')
    args = parser.parse_args()

    results = search_word_in_files(word=args.search_word, root_dir=args.base_root_directory, file_type=args.file_type)
    df_results = pd.DataFrame(results)

    df_results.columns = ['file_path', 'line_number', 'line']
    df_results.to_csv(os.path.join(f'search_{args.search_word}.csv'))
    print(df_results)


if __name__ == '__main__':
    main()