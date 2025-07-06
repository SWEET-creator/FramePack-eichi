#!/usr/bin/env python3
"""
最初と最後のフレームについて、上にtrace、下にpaintを結合するスクリプト
"""

import os
import glob
import cv2
import numpy as np
from pathlib import Path

def combine_trace_paint(paint_path, trace_path, output_path):
    """
    trace画像を上、paint画像を下に結合する
    """
    # 画像を読み込み
    paint_img = cv2.imread(paint_path)
    trace_img = cv2.imread(trace_path)
    
    if paint_img is None:
        print(f"Error: Could not load paint image: {paint_path}")
        return False
    
    if trace_img is None:
        print(f"Error: Could not load trace image: {trace_path}")
        return False
    
    # 画像サイズを確認
    print(f"Paint image size: {paint_img.shape}")
    print(f"Trace image size: {trace_img.shape}")
    
    # サイズを合わせる（幅は小さい方に合わせ、高さは2倍）
    height = min(paint_img.shape[0], trace_img.shape[0])
    width = min(paint_img.shape[1], trace_img.shape[1])
    
    # リサイズ
    paint_resized = cv2.resize(paint_img, (width, height))
    trace_resized = cv2.resize(trace_img, (width, height))
    
    # 縦に結合（上：trace、下：paint）
    combined = np.vstack([trace_resized, paint_resized])
    
    # 保存
    cv2.imwrite(output_path, combined)
    print(f"Combined image saved: {output_path}")
    return True

def main():
    # パスの設定
    paint_dir = "/export/nfs/mukaigawa-lab/datasets/IST02/characters/alleta/paint"
    trace_dir = "/export/nfs/mukaigawa-lab/datasets/IST02/characters/alleta/trace"
    output_dir = "/root/FramePack-eichi/combined_output"
    
    # 出力ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)
    
    # paintファイルのリストを取得
    paint_pattern = os.path.join(paint_dir, "IST02_07_043_B_*.png")
    paint_files = sorted(glob.glob(paint_pattern))
    
    if not paint_files:
        print(f"No paint files found with pattern: {paint_pattern}")
        return
    
    print(f"Found {len(paint_files)} paint files")
    
    # 最初と最後のフレームを処理
    frames_to_process = [
        ("first", paint_files[0]),
        ("last", paint_files[-1])
    ]
    
    for frame_type, paint_file in frames_to_process:
        # ファイル名から番号を抽出
        filename = os.path.basename(paint_file)
        frame_number = filename.split('_')[-1].replace('.png', '')
        
        # 対応するtraceファイルのパス
        trace_file = os.path.join(trace_dir, filename)
        
        # 出力ファイルのパス
        output_filename = f"combined_{frame_type}_frame_{frame_number}.png"
        output_file = os.path.join(output_dir, output_filename)
        
        print(f"\nProcessing {frame_type} frame:")
        print(f"  Paint: {paint_file}")
        print(f"  Trace: {trace_file}")
        print(f"  Output: {output_file}")
        
        # traceファイルの存在確認
        if not os.path.exists(trace_file):
            print(f"  Warning: Trace file not found: {trace_file}")
            continue
        
        # 結合処理
        success = combine_trace_paint(paint_file, trace_file, output_file)
        if success:
            print(f"  ✓ Successfully combined {frame_type} frame")
        else:
            print(f"  ✗ Failed to combine {frame_type} frame")

if __name__ == "__main__":
    main()
