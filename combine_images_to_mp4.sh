#!/bin/bash
# PNG画像を結合してMP4動画を作成するスクリプト
# Usage: ./combine_images_to_mp4.sh

set -e

# 入力画像のパス
IMAGE1="/export/nfs/mukaigawa-lab/datasets/IST02/characters/alleta/trace/IST02_07_043_B_0001.png"
IMAGE2="/export/nfs/mukaigawa-lab/datasets/IST02/characters/alleta/trace/IST02_07_043_B_0002.png"
IMAGE3="/export/nfs/mukaigawa-lab/datasets/IST02/characters/alleta/trace/IST02_07_043_B_0003.png"

# 出力ファイル名（現在のディレクトリに作成）
OUTPUT_DIR="$(pwd)"
OUTPUT_FILE="$OUTPUT_DIR/IST02_07_043_B_combined.mp4"
TEMP_DIR="$OUTPUT_DIR/temp_frames"

echo "=== PNG画像結合 → MP4動画作成スクリプト ==="
echo "出力ディレクトリ: $OUTPUT_DIR"
echo "出力ファイル: $OUTPUT_FILE"

# 入力ファイルの存在確認
echo "入力ファイルの確認中..."
for img in "$IMAGE1" "$IMAGE2" "$IMAGE3"; do
    if [ ! -f "$img" ]; then
        echo "エラー: ファイルが見つかりません: $img"
        exit 1
    else
        echo "✓ 確認済み: $(basename "$img")"
    fi
done

# 一時ディレクトリの作成
mkdir -p "$TEMP_DIR"

# 画像を一時ディレクトリにコピー（連番ファイル名で）
echo "画像を一時ディレクトリにコピー中..."
cp "$IMAGE1" "$TEMP_DIR/frame_001.png"
cp "$IMAGE2" "$TEMP_DIR/frame_002.png"
cp "$IMAGE3" "$TEMP_DIR/frame_003.png"

# ffmpegで動画作成
echo "ffmpegで動画を作成中..."
echo "設定: 各フレーム1秒表示、H.264エンコード、品質CRF=23"

ffmpeg -y \
    -framerate 3 \
    -i "$TEMP_DIR/frame_%03d.png" \
    -c:v libx264 \
    -preset medium \
    -crf 23 \
    -pix_fmt yuv420p \
    -t 3 \
    "$OUTPUT_FILE"

# 一時ディレクトリの削除
echo "一時ファイルをクリーンアップ中..."
rm -rf "$TEMP_DIR"

# 結果の確認
if [ -f "$OUTPUT_FILE" ]; then
    echo "✅ 動画作成完了!"
    echo "出力ファイル: $OUTPUT_FILE"
    echo "ファイルサイズ: $(du -h "$OUTPUT_FILE" | cut -f1)"
    
    # 動画情報を表示
    echo ""
    echo "=== 動画情報 ==="
    ffprobe -v quiet -print_format json -show_format -show_streams "$OUTPUT_FILE" | \
        python3 -c "
import json, sys
data = json.load(sys.stdin)
format_info = data.get('format', {})
video_stream = next((s for s in data.get('streams', []) if s.get('codec_type') == 'video'), {})

print(f'時間: {float(format_info.get(\"duration\", 0)):.2f}秒')
print(f'解像度: {video_stream.get(\"width\", \"不明\")}x{video_stream.get(\"height\", \"不明\")}')
print(f'フレームレート: {video_stream.get(\"r_frame_rate\", \"不明\")}')
print(f'コーデック: {video_stream.get(\"codec_name\", \"不明\")}')
"
else
    echo "❌ エラー: 動画作成に失敗しました"
    exit 1
fi

echo ""
echo "🎬 スクリプト実行完了!"
