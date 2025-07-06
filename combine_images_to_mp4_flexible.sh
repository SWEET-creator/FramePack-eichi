#!/bin/bash
# PNG画像を結合してMP4動画を作成するスクリプト（汎用版）
# Usage: ./combine_images_to_mp4_flexible.sh [image1] [image2] [image3] [output_name]

set -e

# デフォルトの入力画像パス
DEFAULT_IMAGE1="/export/nfs/mukaigawa-lab/datasets/IST02/characters/alleta/trace/IST02_07_043_B_0001.png"
DEFAULT_IMAGE2="/export/nfs/mukaigawa-lab/datasets/IST02/characters/alleta/trace/IST02_07_043_B_0002.png"
DEFAULT_IMAGE3="/export/nfs/mukaigawa-lab/datasets/IST02/characters/alleta/trace/IST02_07_043_B_0003.png"
DEFAULT_OUTPUT="IST02_07_043_B_combined"

# コマンドライン引数の処理
IMAGE1="${1:-$DEFAULT_IMAGE1}"
IMAGE2="${2:-$DEFAULT_IMAGE2}"
IMAGE3="${3:-$DEFAULT_IMAGE3}"
OUTPUT_NAME="${4:-$DEFAULT_OUTPUT}"

# 出力設定
OUTPUT_DIR="$(pwd)"
OUTPUT_FILE="$OUTPUT_DIR/${OUTPUT_NAME}.mp4"
TEMP_DIR="$OUTPUT_DIR/temp_frames_$(date +%s)"

# 使用方法の表示
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo "=== PNG画像結合 → MP4動画作成スクリプト（汎用版） ==="
    echo "使用方法:"
    echo "  $0 [image1] [image2] [image3] [output_name]"
    echo ""
    echo "引数:"
    echo "  image1      : 1枚目の画像パス（デフォルト: $DEFAULT_IMAGE1）"
    echo "  image2      : 2枚目の画像パス（デフォルト: $DEFAULT_IMAGE2）"
    echo "  image3      : 3枚目の画像パス（デフォルト: $DEFAULT_IMAGE3）"
    echo "  output_name : 出力ファイル名（拡張子なし、デフォルト: $DEFAULT_OUTPUT）"
    echo ""
    echo "例:"
    echo "  $0                                    # デフォルト設定で実行"
    echo "  $0 img1.png img2.png img3.png my_video # カスタム画像で実行"
    exit 0
fi

echo "=== PNG画像結合 → MP4動画作成スクリプト（汎用版） ==="
echo "入力画像1: $IMAGE1"
echo "入力画像2: $IMAGE2"
echo "入力画像3: $IMAGE3"
echo "出力ファイル: $OUTPUT_FILE"

# 入力ファイルの存在確認
echo ""
echo "入力ファイルの確認中..."
for i, img in 1 "$IMAGE1" 2 "$IMAGE2" 3 "$IMAGE3"; do
    if [ ! -f "$img" ]; then
        echo "❌ エラー: ファイルが見つかりません: $img"
        exit 1
    else
        echo "✓ 画像$i: $(basename "$img")"
        # 画像サイズも表示
        if command -v identify >/dev/null 2>&1; then
            size=$(identify -format "%wx%h" "$img" 2>/dev/null || echo "不明")
            echo "  サイズ: $size"
        fi
    fi
done

# 一時ディレクトリの作成
mkdir -p "$TEMP_DIR"

# 画像を一時ディレクトリにコピー（連番ファイル名で）
echo ""
echo "画像を一時ディレクトリにコピー中..."
cp "$IMAGE1" "$TEMP_DIR/frame_001.png"
cp "$IMAGE2" "$TEMP_DIR/frame_002.png"
cp "$IMAGE3" "$TEMP_DIR/frame_003.png"

# ffmpegで動画作成
echo ""
echo "ffmpegで動画を作成中..."
echo "設定:"
echo "  - 各フレーム1秒表示（合計3秒）"
echo "  - H.264エンコード"
echo "  - 品質: CRF=23（高品質）"
echo "  - ピクセルフォーマット: yuv420p（互換性重視）"

ffmpeg -y \
    -framerate 1 \
    -i "$TEMP_DIR/frame_%03d.png" \
    -c:v libx264 \
    -preset medium \
    -crf 23 \
    -pix_fmt yuv420p \
    -t 3 \
    "$OUTPUT_FILE" 2>/dev/null

# 一時ディレクトリの削除
echo ""
echo "一時ファイルをクリーンアップ中..."
rm -rf "$TEMP_DIR"

# 結果の確認
if [ -f "$OUTPUT_FILE" ]; then
    echo ""
    echo "✅ 動画作成完了!"
    echo "出力ファイル: $OUTPUT_FILE"
    echo "ファイルサイズ: $(du -h "$OUTPUT_FILE" | cut -f1)"
    
    # 動画情報を表示
    echo ""
    echo "=== 動画情報 ==="
    if command -v ffprobe >/dev/null 2>&1; then
        ffprobe -v quiet -print_format json -show_format -show_streams "$OUTPUT_FILE" 2>/dev/null | \
            python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    format_info = data.get('format', {})
    video_stream = next((s for s in data.get('streams', []) if s.get('codec_type') == 'video'), {})

    duration = float(format_info.get('duration', 0))
    width = video_stream.get('width', '不明')
    height = video_stream.get('height', '不明')
    framerate = video_stream.get('r_frame_rate', '不明')
    codec = video_stream.get('codec_name', '不明')

    print(f'時間: {duration:.2f}秒')
    print(f'解像度: {width}x{height}')
    print(f'フレームレート: {framerate}')
    print(f'コーデック: {codec}')
except:
    print('動画情報の取得に失敗しました')
" 2>/dev/null || echo "動画情報の取得に失敗しました"
    fi
else
    echo ""
    echo "❌ エラー: 動画作成に失敗しました"
    exit 1
fi

echo ""
echo "🎬 スクリプト実行完了!"
echo ""
echo "次のステップ:"
echo "  - 動画再生: mpv '$OUTPUT_FILE' または vlc '$OUTPUT_FILE'"
echo "  - 動画確認: ffplay '$OUTPUT_FILE'"
