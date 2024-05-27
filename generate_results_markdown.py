import argparse
import sys
from pathlib import Path

def read_markdown_table(file_path):
    with file_path.open('r') as file:
        return file.read()

def main(experiments_path: str, id: str):
    experiments_path = Path(experiments_path)
    tables_path = experiments_path / "ToySGD" / "tables"
    
    if not tables_path.exists():
        sys.stderr.write("Error: 'tables' folder does not exist. Run 'generate_tables.py' first.\n")
        sys.exit(1)

    statistics_types = ['iqm', 'mean', 'lowest']
    functions = ['Ackley', 'Rastrigin', 'Rosenbrock', 'Sphere']
    section_titles = {
        'iqm': 'IQM',
        'mean': 'Mean ± Std',
        'lowest': 'Lowest'
    }

    experiment_folder_name = experiments_path.name
    output_file_path = experiments_path / f"{experiment_folder_name}_{id}.md"

    with output_file_path.open('w') as output_file:
        output_file.write(f"# {experiment_folder_name}\n")
        
        for stat_type in statistics_types:
            output_file.write(f"## {section_titles[stat_type]}\n")
            for func in functions:
                file_name = f"{stat_type}_{func}_{id}.md"
                file_path = tables_path / file_name
                if file_path.exists():
                    table_content = read_markdown_table(file_path)
                    output_file.write(f"### {func}\n")
                    output_file.write(table_content + "\n")
                else:
                    sys.stderr.write(f"Warning: File {file_name} does not exist in 'tables' folder.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile experiment results into a markdown file.")
    parser.add_argument("experiments_path", type=str, help="Path to the experiments folder.")
    parser.add_argument("id", type=str, help="ID of teacher trained on.", default=0)
    args = parser.parse_args()
    
    main(args.experiments_path, args.id)
