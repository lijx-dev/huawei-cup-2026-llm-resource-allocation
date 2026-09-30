"""问题三输入输出路径，均相对于仓库根目录。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
A_DIR = ROOT / 'data/real_attachments/A_data_value/regmix_tables'
B_DIR = ROOT / 'data/real_attachments/B_scaling_laws'
C7 = ROOT / 'data/real_attachments/C_efficiency_evolution/model_architecture_metadata.csv'
OUTPUT_DIR = ROOT / 'results/q3'
INPUT_DIR = OUTPUT_DIR / 'inputs'
