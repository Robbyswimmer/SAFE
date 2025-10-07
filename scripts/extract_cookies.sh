#!/bin/bash
# Extract YouTube cookies from browser using yt-dlp
# Run this on your LOCAL machine (not cluster)

echo "YouTube Cookie Extractor"
echo "========================"
echo ""
echo "This will extract cookies from your browser for YouTube."
echo "You must be logged into YouTube in your browser."
echo ""

# Check if yt-dlp is installed
if ! command -v yt-dlp &> /dev/null; then
    echo "❌ yt-dlp not found. Install with: pip install yt-dlp"
    exit 1
fi

# Detect browser
echo "Select your browser:"
echo "1) Chrome"
echo "2) Firefox"
echo "3) Safari"
echo "4) Edge"
read -p "Enter choice (1-4): " choice

case $choice in
    1) BROWSER="chrome" ;;
    2) BROWSER="firefox" ;;
    3) BROWSER="safari" ;;
    4) BROWSER="edge" ;;
    *) echo "❌ Invalid choice"; exit 1 ;;
esac

OUTPUT_FILE="scripts/cookies.txt"

echo ""
echo "🔧 Extracting cookies from $BROWSER..."
echo ""

# Use yt-dlp to extract cookies
yt-dlp --cookies-from-browser "$BROWSER" \
       --cookies "$OUTPUT_FILE" \
       --skip-download \
       "https://www.youtube.com/watch?v=dQw4w9WgXcQ" 2>/dev/null

if [ -f "$OUTPUT_FILE" ]; then
    echo "✅ Cookies extracted successfully!"
    echo "   Saved to: $OUTPUT_FILE"
    echo ""
    echo "📤 Next steps:"
    echo "   1. Upload to cluster:"
    echo "      scp $OUTPUT_FILE rmose009@bcc:~/SAFE/scripts/cookies.txt"
    echo ""
    echo "   2. Run on cluster:"
    echo "      python scripts/download_and_process_audiocaps.py \\"
    echo "        --split train --append \\"
    echo "        --cookies ./scripts/cookies.txt \\"
    echo "        --num-workers 2"
else
    echo "❌ Failed to extract cookies"
    echo ""
    echo "Manual alternative:"
    echo "1. Install browser extension: 'Get cookies.txt LOCALLY'"
    echo "2. Go to youtube.com (while logged in)"
    echo "3. Click extension and export cookies"
    echo "4. Save as $OUTPUT_FILE"
fi
