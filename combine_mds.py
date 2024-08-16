from pathlib import Path

def find_files(root_dir, file_name):
    """Find all files named file_name in the root_dir without searching subdirectories."""
    root_path = Path(root_dir)
    return sorted(root_path.rglob(file_name))

def read_and_concatenate(files):
    """Read and concatenate the content of the files, keeping the first 2 rows of the first file only."""
    concatenated_content = ""
    for i, file in enumerate(files):
        with file.open('r') as f:
            lines = f.readlines()
            if i == 0:
                # Keep all lines of the first file
                concatenated_content += "".join(lines)
            else:
                # Skip the first 2 lines for other files
                concatenated_content += "".join(lines[2:])
            # Add a new line after each file
            concatenated_content += "\n"
    return concatenated_content

def main(root_dir):
    for file_prefix in ["mean", "iqm", "lowest"]:
        file_name = f'tables/{file_prefix}_agent_0.md'
        files = find_files(root_dir, file_name)
        if not files:
            print("No files found.")
            return

        concatenated_content = read_and_concatenate(files)

        # Save the concatenated content to a new file
        output_path = Path(f'concatenated_{file_prefix}_agent_0.md')
        output_path.write_text(concatenated_content)
        print("Concatenated content saved to concatenated_mean_agent_0.md")

if __name__ == "__main__":
    root_dir = 'data_hetero_256_mixed_60k/ToySGD'  # Replace with your root directory
    main(root_dir)
