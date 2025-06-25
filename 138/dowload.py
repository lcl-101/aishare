
# Take SeedVR2-3B as an example.
# See all models: https://huggingface.co/models?other=seedvr

from huggingface_hub import snapshot_download

save_dir = "ckpts/"
repo_id = "ByteDance-Seed/SeedVR2-3B"
cache_dir = save_dir + "/cache"

snapshot_download(cache_dir=cache_dir,
  local_dir=save_dir,
  repo_id=repo_id,
  local_dir_use_symlinks=False,
  resume_download=True,
  allow_patterns=["*.json", "*.safetensors", "*.pth", "*.bin", "*.py", "*.md", "*.txt"],
)

save_dir = "ckpts/"
repo_id = "ByteDance-Seed/SeedVR2-7B"
cache_dir = save_dir + "/cache"

snapshot_download(cache_dir=cache_dir,
  local_dir=save_dir,
  repo_id=repo_id,
  local_dir_use_symlinks=False,
  resume_download=True,
  allow_patterns=["*.json", "*.safetensors", "*.pth", "*.bin", "*.py", "*.md", "*.txt"],
)


repo_id = "Iceclear/SeedVR_VideoDemos"
local_directory = "./demo_video # 你希望保存的本地路径

print(f"开始下载 {repo_id} 到 {local_directory}...")

# snapshot_download 会下载整个仓库
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=local_directory,
    local_dir_use_symlinks=False # 建议设为False，直接复制文件而不是创建符号链接
)

print("下载完成！")