# llm-benchmark

## Before start

### 0. git clone this repo (llm-benchmark)

```bash
git clone https://github.com/AaronZLT/llm-benchmark.git
```

### 1. install llmbenchmark

```bash
# (option) conda & pypi mirror site
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# install llmbenchmark
cd llm-benchmark
pip install .

# install flash-attn
pip install flash-attn --no-build-isolation
```

## Examples

### 2.1 from llm-benchmark/examples
```bash
cd llm-benchmark/examples/Benchmark
```
Modify LLM_BENCHMARK_PATH in the cmds.sh to the path to llm-benchmark, then:
```bash
./cmds.sh
```

### 2.2 from your code
```python
import llmbenchmark
llmbenchmark.train()
```
