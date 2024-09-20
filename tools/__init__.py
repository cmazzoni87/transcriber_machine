from pathlib import Path
home = Path(__file__).parent.parent
RUST_PARSE_TOOL = home / 'rusty_tools' / 'text_splitter_rs' / 'target' / 'release' / 'text_splitter_rs'
RUST_TXT_TOOL = home / 'rusty_tools' / 'text_parser_rs' / 'target' / 'release' / 'text_parser_rs'
DATA_DIR = home / 'output_results'


