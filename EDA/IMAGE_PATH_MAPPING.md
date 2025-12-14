# Image Path Mapping for Q1 Generation

## Path Mismatch Issue

After copying the codebase from VM, image paths in QA files don't match actual file locations.

### Original VM Paths (in QA JSON files)
```
/home/himanshu/dev/test/extracted_images/renders/
```

### Actual Current Paths
```
/home/himanshu/MSC_FINAL/dev/test/test/extracted_images/renders/
```

## Solution Options

### Option 1: Update QA Files (Recommended)

Update all `image_path` fields in QA JSON files:

```bash
cd /home/himanshu/MSC_FINAL/dev/test/test/data/processed/qa_pairs_individual_components

# Backup first
cp -r . ../qa_pairs_individual_components_backup

# Update paths
find . -name "*.json" -type f -exec sed -i 's|/home/himanshu/dev/test/extracted_images|/home/himanshu/MSC_FINAL/dev/test/test/extracted_images|g' {} \;
```

### Option 2: Create Symlink

Create a symlink to match the old path:

```bash
sudo mkdir -p /home/himanshu/dev/test
sudo ln -s /home/himanshu/MSC_FINAL/dev/test/test/extracted_images /home/himanshu/dev/test/extracted_images
```

### Option 3: Update Code with Path Fallback

Modify `load_image_sanitized()` in `multimodal_qa_runner_vllm.py` to try multiple paths:

```python
def load_image_sanitized(image_path: str) -> Optional[Image.Image]:
    """Load and sanitize image (remove EXIF, re-encode) with path fallback."""
    p = Path(image_path)
    
    # Try original path first
    if p.exists():
        return _load_and_sanitize(p)
    
    # Try path with MSC_FINAL prefix
    if '/home/himanshu/dev/' in str(p):
        fallback_path = str(p).replace('/home/himanshu/dev/', '/home/himanshu/MSC_FINAL/dev/')
        p_fallback = Path(fallback_path)
        if p_fallback.exists():
            return _load_and_sanitize(p_fallback)
    
    # Try with test/test/ addition
    if '/test/extracted_images' in str(p) and '/test/test/extracted_images' not in str(p):
        fallback_path = str(p).replace('/test/extracted_images', '/test/test/extracted_images')
        p_fallback = Path(fallback_path)
        if p_fallback.exists():
            return _load_and_sanitize(p_fallback)
    
    return None
```

## Verification

Check if images can be loaded:

```bash
# Check one QA file
QA_FILE="dev/test/test/data/processed/qa_pairs_individual_components/3_22'-Dichlorodiethyl_Sulfide_Mustard_Gas.json"
IMAGE_PATH=$(cat "$QA_FILE" | python3 -c "import json, sys; print(json.load(sys.stdin).get('image_path', ''))")

echo "Path in QA file: $IMAGE_PATH"

# Check if exists at original path
if [ -f "$IMAGE_PATH" ]; then
    echo "✅ Image found at original path"
else
    echo "❌ Image not found at original path"
    
    # Try to find actual location
    IMAGE_NAME=$(basename "$IMAGE_PATH")
    ACTUAL_PATH=$(find /home/himanshu/MSC_FINAL/dev -name "$IMAGE_NAME" 2>/dev/null | head -1)
    
    if [ -n "$ACTUAL_PATH" ]; then
        echo "✅ Image found at: $ACTUAL_PATH"
        echo "   Update QA file to use this path"
    else
        echo "❌ Image not found anywhere"
    fi
fi
```

## Summary

- **QA files reference:** `/home/himanshu/dev/test/extracted_images/renders/`
- **Actual location:** `/home/himanshu/MSC_FINAL/dev/test/test/extracted_images/renders/`
- **Fix:** Update QA files OR create symlink OR add path fallback in code

