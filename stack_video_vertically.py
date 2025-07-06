#!/usr/bin/env python3
"""
動画を上下に積み重ねて保存するスクリプト
同じ動画を上下に重ねて、縦に2倍の高さの動画を作成します
"""

import os
import subprocess
import sys
from pathlib import Path

def stack_video_vertically(input_path, output_path=None):
    """
    動画を上下に積み重ねて保存する
    
    Args:
        input_path (str): 入力動画のパス
        output_path (str): 出力動画のパス（省略時は自動生成）
    
    Returns:
        bool: 成功時True、失敗時False
    """
    input_path = Path(input_path)
    
    # 入力ファイルの存在確認
    if not input_path.exists():
        print(f"Error: Input file does not exist: {input_path}")
        return False
    
    # 出力パスが指定されていない場合は自動生成
    if output_path is None:
        stem = input_path.stem
        suffix = input_path.suffix
        output_path = input_path.parent / f"{stem}_doubled_height{suffix}"
    else:
        output_path = Path(output_path)
    
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    
    # 音声ストリームの存在を確認
    probe_cmd = [
        "ffprobe", "-v", "quiet", "-select_streams", "a", 
        "-show_entries", "stream=index", "-of", "csv=p=0", str(input_path)
    ]
    
    try:
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
        has_audio = bool(probe_result.stdout.strip())
    except:
        has_audio = False
    
    # FFmpegコマンドを構築
    # vstack filter を使用して同じ動画を上下に積み重ね
    cmd = [
        "ffmpeg",
        "-i", str(input_path),      # 入力動画（上部用）
        "-i", str(input_path),      # 入力動画（下部用）
        "-filter_complex", "[0:v][1:v]vstack=inputs=2[v]",  # 上下に積み重ね
        "-map", "[v]",              # 映像ストリームをマップ
        "-c:v", "libx264",          # H.264エンコーダ使用
        "-preset", "medium",        # エンコード品質設定
        "-crf", "23",               # 品質設定（0-51、低いほど高品質）
        "-y"                        # 出力ファイルを上書き
    ]
    
    # 音声がある場合のみ音声関連のオプションを追加
    if has_audio:
        cmd.extend(["-map", "0:a", "-c:a", "copy"])
        print("Audio stream detected - will be copied to output")
    else:
        print("No audio stream detected - creating video-only output")
    
    cmd.append(str(output_path))
    
    try:
        print("Processing video...")
        print(f"Command: {' '.join(cmd)}")
        
        # FFmpegを実行
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        print("Video processing completed successfully!")
        print(f"Output saved to: {output_path}")
        
        # 出力ファイルサイズを確認
        if output_path.exists():
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"Output file size: {size_mb:.1f} MB")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error during video processing:")
        print(f"Return code: {e.returncode}")
        print(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python stack_video_vertically.py <input_video> [output_video]")
        print("Example: python stack_video_vertically.py input.mp4")
        print("Example: python stack_video_vertically.py input.mp4 output_stacked.mp4")
        sys.exit(1)
    
    input_video = sys.argv[1]
    output_video = sys.argv[2] if len(sys.argv) > 2 else None
    
    success = stack_video_vertically(input_video, output_video)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
