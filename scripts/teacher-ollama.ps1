# Private Ollama instance for the teacher: own port, 2 parallel slots, models from D:\OllamaModels.
# Leaves the user's main Ollama (:11434, 262K default context) untouched.
$env:OLLAMA_HOST = "127.0.0.1:11435"
$env:OLLAMA_NUM_PARALLEL = "2"
$env:OLLAMA_CONTEXT_LENGTH = "5632"
$env:OLLAMA_MAX_LOADED_MODELS = "1"
$env:OLLAMA_KEEP_ALIVE = "30m"
$env:OLLAMA_MODELS = "D:\OllamaModels"
$env:OLLAMA_FLASH_ATTENTION = "1"
$env:OLLAMA_KV_CACHE_TYPE = "q8_0"
ollama serve *> D:\distill-lab\work\teacher-ollama.log
