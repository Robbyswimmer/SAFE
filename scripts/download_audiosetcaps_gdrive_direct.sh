#!/bin/bash
# Download AudioSetCaps audio files directly from Google Drive with robustness

# Do NOT exit on error - we want to continue downloading other files
set +e

OUTPUT_DIR=${1:-"data/audiosetcaps"}
TEMP_DIR="${OUTPUT_DIR}_temp"
LOG_FILE="${TEMP_DIR}/download.log"
FAILED_FILE="${TEMP_DIR}/failed_downloads.txt"
SUCCESS_FILE="${TEMP_DIR}/successful_downloads.txt"

echo "Installing gdown if needed..."
pip install -q gdown

mkdir -p "$OUTPUT_DIR"
mkdir -p "$TEMP_DIR"

# Ensure we're in temp dir for log files
cd "$TEMP_DIR"

# Initialize log files
touch "download.log"
touch "failed_downloads.txt"
touch "successful_downloads.txt"

# Set absolute paths
LOG_FILE="$(pwd)/download.log"
FAILED_FILE="$(pwd)/failed_downloads.txt"
SUCCESS_FILE="$(pwd)/successful_downloads.txt"

echo "=== AudioSetCaps Download ===" | tee -a "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"
echo "Folder: https://drive.google.com/drive/folders/1ZKyRZw3AhS3HkWivgMqtMODB0TkVPNk5" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

echo "Attempting to download folder (errors will be logged and skipped)..." | tee -a "$LOG_FILE"
echo "Progress will continue even if some files fail." | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Count files before download
before_count=$(find . -type f \( -name "*.tar" -o -name "*.tar.gz" \) 2>/dev/null | wc -l)
echo "Files before download: $before_count" | tee -a "$LOG_FILE"

# Run gdown multiple times to ensure we get all files
# Sometimes gdown stops early, so we retry until no new files are downloaded
max_attempts=5
attempt=1

while [ $attempt -le $max_attempts ]; do
    echo "" | tee -a "$LOG_FILE"
    echo "Download attempt $attempt of $max_attempts..." | tee -a "$LOG_FILE"

    # Run gdown with --remaining-ok to skip already downloaded files
    if gdown --folder https://drive.google.com/drive/folders/1ZKyRZw3AhS3HkWivgMqtMODB0TkVPNk5 --remaining-ok 2>&1 | tee -a "$LOG_FILE"; then
        echo "Download attempt $attempt completed" | tee -a "$LOG_FILE"
    else
        echo "Download attempt $attempt finished with some errors (continuing anyway)" | tee -a "$LOG_FILE"
    fi

    # Count files after this attempt
    after_count=$(find . -type f \( -name "*.tar" -o -name "*.tar.gz" \) 2>/dev/null | wc -l)
    new_files=$((after_count - before_count))

    echo "Files downloaded in attempt $attempt: $new_files (total: $after_count)" | tee -a "$LOG_FILE"

    # If no new files were downloaded, we're done
    if [ $new_files -eq 0 ]; then
        echo "No new files downloaded, stopping" | tee -a "$LOG_FILE"
        break
    fi

    before_count=$after_count
    attempt=$((attempt + 1))

    # Small delay between attempts
    sleep 2
done

echo "" | tee -a "$LOG_FILE"
echo "=== Download Summary ===" | tee -a "$LOG_FILE"

# Count downloaded files
downloaded_count=$(find . -type f \( -name "*.tar" -o -name "*.tar.gz" \) | wc -l)
echo "Downloaded tar files: $downloaded_count" | tee -a "$LOG_FILE"

# Extract all tar files with error handling
echo "" | tee -a "$LOG_FILE"
echo "=== Extracting tar files ===" | tee -a "$LOG_FILE"

extracted_count=0
failed_extract_count=0

# Process .tar files
shopt -s nullglob  # Make globs expand to nothing if no matches
for tarfile in *.tar *.tar.gz; do
    # Skip if no files match (glob didn't expand)
    [ -f "$tarfile" ] || continue

    echo "Extracting $tarfile..." | tee -a "$LOG_FILE"

    # Try to extract
    if tar -xf "$tarfile" -C "../${OUTPUT_DIR#*/}" 2>&1 | tee -a "$LOG_FILE"; then
        echo "✓ Extracted $tarfile" | tee -a "$LOG_FILE"
        echo "$tarfile" >> "$SUCCESS_FILE"
        ((extracted_count++))

        # Optionally remove tar file to save space after successful extraction
        # Uncomment the next line to auto-delete after extraction:
        # rm "$tarfile" && echo "  Deleted $tarfile to save space" | tee -a "$LOG_FILE"
    else
        echo "✗ Failed to extract $tarfile" | tee -a "$LOG_FILE"
        echo "$tarfile" >> "$FAILED_FILE"
        ((failed_extract_count++))
    fi

    echo "" | tee -a "$LOG_FILE"
done

echo "=== Final Summary ===" | tee -a "$LOG_FILE"
echo "Downloaded files: $downloaded_count" | tee -a "$LOG_FILE"
echo "Successfully extracted: $extracted_count" | tee -a "$LOG_FILE"
echo "Failed to extract: $failed_extract_count" | tee -a "$LOG_FILE"
echo "Completed: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

if [ $failed_extract_count -gt 0 ]; then
    echo "⚠️  Some files failed to extract. See $FAILED_FILE for details." | tee -a "$LOG_FILE"
fi

echo "✓ Process complete!" | tee -a "$LOG_FILE"
echo "Audio files in: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "Tar files in: $TEMP_DIR" | tee -a "$LOG_FILE"
echo "Full log: $LOG_FILE" | tee -a "$LOG_FILE"
echo "Successfully extracted: $SUCCESS_FILE" | tee -a "$LOG_FILE"

if [ $failed_extract_count -gt 0 ]; then
    echo "Failed files: $FAILED_FILE" | tee -a "$LOG_FILE"
    exit 1
fi

exit 0
