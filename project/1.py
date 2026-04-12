import shutil

src = "/LimYeonGyeong/project/paged_llama/llama/modeling/modeling_llama.py"
dst = "/usr/local/lib/python3.10/dist-packages/transformers/models/llama/modeling_llama.py"
bak = "/usr/local/lib/python3.10/dist-packages/transformers/models/llama/modeling_llama_backup.py"

# 1. 원본 백업
shutil.copy2(dst, bak)

# 2. 네가 수정한 파일로 덮어쓰기
shutil.copy2(src, dst)

print("✅ 복사 완료")
