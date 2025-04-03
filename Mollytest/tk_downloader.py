from transformers import AutoTokenizer, AutoModel
import os
import tarfile

def download_model_and_tokenizer(model_name, output_dir, export_tar=True):
    print(f"Downloading model and tokenizer for: {model_name}")
    
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(output_dir)

    model = AutoModel.from_pretrained(model_name)
    model.save_pretrained(output_dir)

    print(f"All model files saved to: {output_dir}")

    if export_tar:
        tar_path = output_dir.rstrip('/').rstrip('\\') + ".tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(output_dir, arcname=os.path.basename(output_dir))
        print(f"Tarball created: {tar_path}")

if __name__ == "__main__":
    model_name = "Rostlab/prot_bert_bfd"
    output_dir = "./prot_bert_bfd_tokenizer"
    download_model_and_tokenizer(model_name, output_dir)

"""
from transformers import AutoTokenizer
import os
import shutil
import tarfile

def download_and_export_tokenizer(model_name, output_dir, export_tar=True):
    print(f"Downloading tokenizer for model: {model_name}")
    
    # 確保輸出資料夾存在
    os.makedirs(output_dir, exist_ok=True)

    # 從 Hugging Face 下載 tokenizer,直接存到指定資料夾
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(output_dir)
    print(f"Tokenizer saved to: {output_dir}")

    # 壓縮成 .tar.gz
    if export_tar:
        tar_path = output_dir.rstrip('/').rstrip('\\') + ".tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(output_dir, arcname=os.path.basename(output_dir))
        print(f"Tarball created: {tar_path}")

# ============ 使用範例 =============
if __name__ == "__main__":
    model_name = "Rostlab/prot_bert_bfd"
    output_dir = "./prot_bert_bfd_tokenizer"

    download_and_export_tokenizer(model_name, output_dir)
"""