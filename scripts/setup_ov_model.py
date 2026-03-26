"""
setup_ov_model.py - 从 HuggingFace 下载 OpenVINO/Qwen2.5-VL-7B-Instruct-int4-ov，
                    并部署到 <SKILL_DIR>/models/ 目录下。

输出路径：<SKILL_DIR>/models/<OUTPUT_DIR_NAME>/
  - 默认输出子目录名 = Qwen2.5-VL-7B-Instruct-int4

用法：
    # 基础（默认使用 hf-mirror 镜像，国内推荐）
    python setup_ov_model.py

    # 使用自定义 HF 镜像地址
    python setup_ov_model.py --hf-mirror https://hf-mirror.com

    # 不使用镜像（直连 HuggingFace，需要科学上网）
    python setup_ov_model.py --no-mirror

    # 指定代理（国内网络）
    set HTTPS_PROXY=http://127.0.0.1:7890
    python setup_ov_model.py

    # 指定模型目录
    python setup_ov_model.py --model-dir D:\\path\\to\\models\\Qwen2.5-VL-7B-Instruct-int4

    # 强制重新下载（即使目录已存在）
    python setup_ov_model.py --force

    # 只校验已有模型是否存在，不执行下载
    python setup_ov_model.py --check-only

依赖（运行前需安装）：
    pip install huggingface_hub
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 路径常量
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_DIR = SCRIPT_DIR.parent
MODELS_DIR = SKILL_DIR / "models"

# HuggingFace 模型 ID（已转换好的 OpenVINO INT4 模型）
HF_MODEL_ID = "OpenVINO/Qwen2.5-VL-7B-Instruct-int4-ov"

# 默认输出子目录名
DEFAULT_OUTPUT_NAME = "Qwen2.5-VL-7B-Instruct-int4"

# HF 镜像站地址（国内网络推荐使用）
HF_MIRROR_URL = "https://hf-mirror.com"

# 虚拟环境所需依赖（最小集合）
VENV_PACKAGES = ["huggingface_hub"]


# ---------------------------------------------------------------------------
# 虚拟环境管理
# ---------------------------------------------------------------------------

def _get_venv_python() -> "Path | None":
    """返回 <SKILL_DIR>/.venv 中的 Python 可执行路径，不存在返回 None。"""
    candidate = SKILL_DIR / ".venv" / "Scripts" / "python.exe"
    return candidate if candidate.exists() else None


def _ensure_venv(packages: list) -> None:
    """确保 <SKILL_DIR>/.venv 存在并安装了指定包列表。"""
    venv_dir = SKILL_DIR / ".venv"
    venv_python = _get_venv_python()

    if venv_python is None:
        print(f"[venv] 虚拟环境不存在，正在创建：{venv_dir}")
        subprocess.run(
            [sys.executable, "-m", "venv", str(venv_dir)],
            check=True,
        )
        venv_python = _get_venv_python()
        if venv_python is None:
            raise RuntimeError(f"创建虚拟环境后仍找不到 Python 可执行文件：{venv_dir}")
        print(f"[venv] ✓ 虚拟环境已创建：{venv_dir}")
    else:
        print(f"[venv] 已找到虚拟环境：{venv_python}")

    if packages:
        print(f"[venv] 安装依赖：{', '.join(packages)}")
        subprocess.run(
            [str(venv_python), "-m", "pip", "install", "--quiet", *packages],
            check=True,
        )
        print(f"[venv] ✓ 依赖安装完成")


# ---------------------------------------------------------------------------
# 模型目录检查
# ---------------------------------------------------------------------------

# OpenVINO 模型目录完整性阈值
MODEL_MIN_XML_FILES    = 1
MODEL_MIN_BIN_FILES    = 1
MODEL_MIN_TOTAL_ENTRIES = 27


def _inspect_model_dir(model_dir: Path) -> dict:
    """
    检查 OpenVINO 模型目录的完整性，返回详细报告。
    """
    if not model_dir.is_dir():
        return {
            "exists": False, "xml_count": 0, "bin_count": 0,
            "total_entries": 0, "valid": False,
            "reason": f"目录不存在：{model_dir}",
        }

    all_entries  = list(model_dir.rglob("*"))
    all_files    = [e for e in all_entries if e.is_file()]
    xml_files    = [f for f in all_files if f.suffix.lower() == ".xml"]
    bin_files    = [f for f in all_files if f.suffix.lower() == ".bin"]
    total_entries = len(all_entries)
    xml_count    = len(xml_files)
    bin_count    = len(bin_files)

    reasons = []
    if xml_count < MODEL_MIN_XML_FILES:
        reasons.append(f".xml 文件数 {xml_count} < 最低要求 {MODEL_MIN_XML_FILES}")
    if bin_count < MODEL_MIN_BIN_FILES:
        reasons.append(f".bin 文件数 {bin_count} < 最低要求 {MODEL_MIN_BIN_FILES}")
    if total_entries < MODEL_MIN_TOTAL_ENTRIES:
        reasons.append(f"总条目数 {total_entries} < 最低要求 {MODEL_MIN_TOTAL_ENTRIES}（下载不完整）")

    valid = len(reasons) == 0
    return {
        "exists": True,
        "xml_count": xml_count,
        "bin_count": bin_count,
        "total_entries": total_entries,
        "valid": valid,
        "reason": "；".join(reasons) if reasons else "",
    }


def _verify_model_dir(model_dir: Path) -> bool:
    """验证 OpenVINO 模型目录是否完整有效。"""
    return _inspect_model_dir(model_dir)["valid"]


# ---------------------------------------------------------------------------
# 模型下载
# ---------------------------------------------------------------------------

def _download_model(
    repo_id: str,
    output_dir: Path,
    hf_endpoint: str | None = None,
) -> bool:
    """使用 huggingface_hub 将整个仓库快照下载到 output_dir。"""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "[model] 未找到 huggingface_hub，请先安装：\n"
            "  pip install huggingface_hub",
            file=sys.stderr,
        )
        return False

    env_backup: dict[str, str | None] = {}
    if hf_endpoint:
        env_backup["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT")
        os.environ["HF_ENDPOINT"] = hf_endpoint

    try:
        print(f"[model] 正在从 {hf_endpoint or 'https://huggingface.co'} 下载 {repo_id} ...")
        print(f"[model] 目标目录：{output_dir}")
        print("[model] 下载文件较大（约 4~6 GB），请耐心等待。")
        print()

        output_dir.mkdir(parents=True, exist_ok=True)

        snapshot_download(
            repo_id=repo_id,
            local_dir=str(output_dir),
            local_dir_use_symlinks=False,
        )
        return True
    except Exception as e:
        print(f"[model] 下载失败：{e}", file=sys.stderr)
        return False
    finally:
        for key, val in env_backup.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def setup_ov_model(
    model_dir: Path,
    repo_id: str,
    force: bool,
    check_only: bool,
    hf_endpoint: str | None = None,
) -> bool:
    """
    从 HuggingFace 下载 OpenVINO 模型并部署到指定目录。

    Returns:
        True 表示成功（含「已存在跳过」），False 表示失败。
    """
    print(f"  模型目录 : {model_dir}")
    print()

    # 仅校验模式
    if check_only:
        report = _inspect_model_dir(model_dir)
        if report["valid"]:
            print(f"[model] ✓ 模型目录完整有效：{model_dir}")
            print(f"[model]   .xml={report['xml_count']}  .bin={report['bin_count']}  总条目={report['total_entries']}")
            return True
        else:
            print(f"[model] ✗ 模型目录不完整：{model_dir}")
            print(f"[model]   .xml={report['xml_count']}  .bin={report['bin_count']}  总条目={report['total_entries']}")
            if report["reason"]:
                print(f"[model]   原因：{report['reason']}")
            print(f"[model]   建议：python setup_ov_model.py --force")
            return False

    # 检查已有目录的完整性
    if not force:
        report = _inspect_model_dir(model_dir)
        if report["valid"]:
            print(f"[model] 模型已存在且完整，跳过下载。（{model_dir}）")
            print(f"[model]   .xml={report['xml_count']}  .bin={report['bin_count']}  总条目={report['total_entries']}")
            return True
        elif report["exists"]:
            print(f"[model] ⚠ 模型目录存在但不完整（总条目={report['total_entries']} < {MODEL_MIN_TOTAL_ENTRIES}），将清除后重新下载。")
            if report["reason"]:
                print(f"[model]   原因：{report['reason']}")
            shutil.rmtree(model_dir, ignore_errors=True)
            print(f"[model]   已清除不完整目录：{model_dir}")

    if force and model_dir.exists():
        print(f"[model] --force 模式：删除已有目录 {model_dir}")
        shutil.rmtree(model_dir, ignore_errors=True)

    model_dir.parent.mkdir(parents=True, exist_ok=True)

    ok = _download_model(
        repo_id=repo_id,
        output_dir=model_dir,
        hf_endpoint=hf_endpoint,
    )

    if not ok:
        return False

    if not _verify_model_dir(model_dir):
        print(
            f"[model] 下载完成，但输出目录验证失败：{model_dir}\n"
            "  请检查上方日志是否有错误。",
            file=sys.stderr,
        )
        return False

    print(f"\n[model] ✓ 模型下载完成：{model_dir}")
    return True


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从 HuggingFace 下载 OpenVINO VLM 模型到 SKILL_DIR/models/ 目录",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--model-dir",
        dest="model_dir",
        default=None,
        metavar="PATH",
        help=(
            f"模型存放目录。未指定时默认为 <SKILL_DIR>/models/{DEFAULT_OUTPUT_NAME} "
            f"(当前: {MODELS_DIR / DEFAULT_OUTPUT_NAME})"
        ),
    )
    parser.add_argument(
        "--repo-id",
        dest="repo_id",
        default=HF_MODEL_ID,
        metavar="REPO_ID",
        help=f"HuggingFace 仓库 ID（默认：{HF_MODEL_ID}）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新下载（即使目标目录已存在）",
    )
    parser.add_argument(
        "--check-only",
        dest="check_only",
        action="store_true",
        help="仅校验模型目录是否已存在，不执行下载",
    )

    mirror_group = parser.add_mutually_exclusive_group()
    mirror_group.add_argument(
        "--hf-mirror",
        dest="hf_mirror",
        nargs="?",
        const=HF_MIRROR_URL,
        default=HF_MIRROR_URL,
        metavar="URL",
        help=(
            f"使用 HuggingFace 镜像站加速下载（默认启用 {HF_MIRROR_URL}）。"
            f"可指定自定义镜像地址，如 --hf-mirror https://hf-mirror.com。"
        ),
    )
    mirror_group.add_argument(
        "--no-mirror",
        dest="no_mirror",
        action="store_true",
        help="禁用镜像站，直连 HuggingFace（需要科学上网）",
    )

    return parser.parse_args()


def main() -> int:
    try:
        _ensure_venv(VENV_PACKAGES)
    except Exception as exc:
        print(f"[venv] 警告：无法创建/配置虚拟环境，将尝试使用系统环境：{exc}", file=sys.stderr)

    args = parse_args()

    # 确定 HF_ENDPOINT
    if args.no_mirror:
        hf_endpoint: str | None = None
    else:
        hf_endpoint = args.hf_mirror or os.environ.get("HF_ENDPOINT") or HF_MIRROR_URL

    # 确定模型目录
    if args.model_dir:
        model_dir = Path(args.model_dir)
    else:
        model_dir = MODELS_DIR / DEFAULT_OUTPUT_NAME

    print("=" * 60)
    print("OpenVINO 模型下载脚本")
    print(f"仓库 ID    : {args.repo_id}")
    print(f"模型目录   : {model_dir}")
    print(f"下载镜像   : {hf_endpoint or '直连 HuggingFace'}")
    print("=" * 60)
    print()

    ok = setup_ov_model(
        model_dir=model_dir,
        repo_id=args.repo_id,
        force=args.force,
        check_only=args.check_only,
        hf_endpoint=hf_endpoint,
    )

    print()
    if ok:
        print("✓ 完成。")
        return 0
    else:
        print("✗ 失败，请查看上方错误信息。", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
