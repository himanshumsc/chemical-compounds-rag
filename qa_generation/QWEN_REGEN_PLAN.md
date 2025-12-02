# Plan: Regenerate QWEN Answers with vLLM Optimization

## Objective
Create an optimized version of the QA runner to regenerate all 4 answers (Q1-Q4) for all 178 files using:
- Token limit: 500 (matching OpenAI calls)
- vLLM for better performance
- Background execution capability
- Logging and monitoring

## Steps

### Phase 1: Setup and Preparation

#### 1.1 Create Backup/Copy
- **Source**: `/home/himanshu/dev/output/qwen/`
- **Destination**: `/home/himanshu/dev/output/qwen_regenerated/`
- **Action**: Copy all existing answer files to preserve originals
- **Files**: 178 JSON files + logs + summary files

#### 1.2 Copy Script
- **Source**: `/home/himanshu/dev/code/multimodal_qa_runner.py`
- **Destination**: `/home/himanshu/dev/code/multimodal_qa_runner_vllm.py`
- **Purpose**: Modified version with vLLM and new requirements

### Phase 2: Script Modifications

#### 2.1 Key Changes Needed

**A. vLLM Integration**
- Replace `AutoModelForImageTextToText` with vLLM's `LLM` class
- Use vLLM's batch processing capabilities
- Configure vLLM for Qwen2.5-VL-AWQ model
- Handle vision inputs with vLLM (may need special handling)

**B. Token Limit Update**
- Change `max_new_tokens` from 128 to 500
- Update all generation calls

**C. Input Source Change**
- **Current**: Reads from `qa_pairs_individual_components/`
- **New**: Read questions from existing answer files in `output/qwen_regenerated/`
- Extract questions from `answers[].question` field
- Preserve existing file structure

**D. Output Strategy**
- **Current**: Creates new files
- **New**: Update existing files in `output/qwen_regenerated/`
- Replace `answers[].answer` with new generated answers
- Keep `latency_s` and other metadata
- Update timestamps

**E. Background Execution**
- Add nohup support
- Comprehensive logging to file
- Progress tracking
- Process ID tracking

#### 2.2 vLLM Implementation Details

**Model Loading**:
```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="/home/himanshu/dev/models/QWEN_AWQ",
    tensor_parallel_size=1,  # Adjust based on GPU
    dtype="float16",
    trust_remote_code=True,
    # Vision model support may need special config
)
```

**Batch Generation**:
```python
sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=500,
    stop=None,
)
outputs = llm.generate(prompts, sampling_params)
```

**Note**: vLLM vision model support may require additional configuration or may not be fully supported. Need to verify.

### Phase 3: Script Structure

#### 3.1 New Function: `regenerate_from_existing_answers()`
- Read existing answer files
- Extract questions
- Generate new answers with vLLM
- Update answer fields
- Save back to same files

#### 3.2 Processing Flow
1. Load all answer files from `output/qwen_regenerated/`
2. For each file:
   - Extract 4 questions
   - Load image for Q1 (from original QA pairs)
   - Generate answers using vLLM
   - Update answer fields
   - Save updated file
3. Track progress and log everything

#### 3.3 Batch Processing Strategy
- Process files in batches (e.g., 10-20 files per batch)
- Use vLLM's native batching
- Group Q1 (with images) separately from Q2-Q4 (text-only)

### Phase 4: Logging and Monitoring

#### 4.1 Log File Structure
- **Location**: `output/qwen_regenerated/logs/regeneration_YYYYMMDD_HHMMSS.log`
- **Format**: Timestamp, level, file, question number, latency, status
- **Content**: Progress, errors, performance metrics

#### 4.2 Progress Tracking
- Log every file processed
- Log every question generated
- Track total time, average latency
- Save summary JSON at end

#### 4.3 Background Execution
- Use nohup for background execution
- Redirect stdout/stderr to log file
- Save process ID to file
- Create status file for monitoring

### Phase 5: Optimization Considerations

#### 5.1 vLLM Configuration
- **Tensor Parallel**: 1 (single GPU) or more if available
- **Batch Size**: Optimize based on GPU memory
- **Max Model Length**: Set appropriately
- **KV Cache**: Enable for faster inference

#### 5.2 Performance Optimizations
- Keep model loaded in memory
- Batch similar requests together
- Pre-load images for Q1
- Use async processing where possible

#### 5.3 Fallback Strategy
- If vLLM doesn't support vision models, fall back to Transformers
- Check vLLM compatibility first
- Document limitations

### Phase 6: Testing and Validation

#### 6.1 Test Run
- Test with 5-10 files first
- Verify answer quality
- Check token usage (should be ~500 max)
- Validate file updates

#### 6.2 Performance Comparison
- Compare with original run
- Measure speedup from vLLM
- Track memory usage
- Document improvements

### Phase 7: Execution Plan

#### 7.1 Background Execution Command
```bash
cd /home/himanshu/dev/code
source .venv_phi4_req/bin/activate  # or appropriate venv
nohup python multimodal_qa_runner_vllm.py \
    --input-dir ../output/qwen_regenerated \
    --qa-dir ../test/data/processed/qa_pairs_individual_components \
    --max-new-tokens 500 \
    --batch-size 10 \
    > ../output/qwen_regenerated/logs/regeneration_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo $! > ../output/qwen_regenerated/regeneration.pid
```

#### 7.2 Monitoring Commands
```bash
# Check if running
ps aux | grep multimodal_qa_runner_vllm

# Monitor progress
tail -f output/qwen_regenerated/logs/regeneration_*.log

# Check progress count
grep "Processing.*/" output/qwen_regenerated/logs/regeneration_*.log | wc -l

# Check for errors
grep -i "error\|failed" output/qwen_regenerated/logs/regeneration_*.log

# Check process status
cat output/qwen_regenerated/regeneration.pid
```

## Implementation Checklist

### Setup
- [ ] Copy `output/qwen/` to `output/qwen_regenerated/`
- [ ] Copy `multimodal_qa_runner.py` to `multimodal_qa_runner_vllm.py`
- [ ] Install vLLM (if not already installed)
- [ ] Verify vLLM supports Qwen2.5-VL-AWQ

### Script Development
- [ ] Integrate vLLM for text-only questions (Q2-Q4)
- [ ] Handle vision questions (Q1) - may need Transformers fallback
- [ ] Update max_new_tokens to 500
- [ ] Modify to read from existing answer files
- [ ] Implement answer replacement logic
- [ ] Add comprehensive logging
- [ ] Add progress tracking
- [ ] Add error handling

### Testing
- [ ] Test with 5 files
- [ ] Verify answer quality
- [ ] Check token usage
- [ ] Validate file updates
- [ ] Test background execution

### Documentation
- [ ] Document vLLM configuration
- [ ] Document execution commands
- [ ] Document monitoring procedures
- [ ] Document fallback strategy

## Potential Challenges

### 1. vLLM Vision Model Support
- **Issue**: vLLM may not support Qwen2.5-VL vision models
- **Solution**: Use vLLM for Q2-Q4, Transformers for Q1
- **Alternative**: Use Transformers for all if vLLM doesn't support vision

### 2. Model Format Compatibility
- **Issue**: vLLM may need different model format
- **Solution**: Check vLLM documentation for AWQ support
- **Alternative**: Convert model or use Transformers

### 3. Batch Processing with Mixed Types
- **Issue**: Q1 has images, Q2-Q4 are text-only
- **Solution**: Process in separate batches
- **Optimization**: Group all Q1s together, all Q2-Q4s together

### 4. Memory Constraints
- **Issue**: vLLM may use more memory
- **Solution**: Adjust batch size, use tensor parallelism
- **Monitoring**: Track GPU memory usage

## Expected Outcomes

### Performance Improvements
- **Current**: 3.57 seconds/question (Transformers)
- **Target**: 2.0-2.5 seconds/question (vLLM)
- **Improvement**: 30-45% faster
- **Total Time**: ~24-30 minutes (vs 42 minutes)

### Quality Improvements
- **Token Limit**: 500 (vs 128)
- **Answer Length**: 2-3x longer answers
- **Completeness**: More comprehensive responses

### Monitoring
- Real-time progress tracking
- Detailed logs
- Error reporting
- Performance metrics

## Next Steps

1. **Review this plan**
2. **Verify vLLM compatibility** with Qwen2.5-VL-AWQ
3. **Create copies** of files
4. **Implement script modifications**
5. **Test with small sample**
6. **Run full regeneration**

