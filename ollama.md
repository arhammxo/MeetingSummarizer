# Setting up Ollama

## 1. Install Ollama

### Linux
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### macOS
```bash
brew install ollama
```

### Windows
Download and install from [https://ollama.com/download/windows](https://ollama.com/download/windows)

## 2. Start Ollama Server
```bash
ollama serve
```

## 3. Pull Recommended Models

For general summarization (choose one based on your hardware capabilities):
```bash
# Best performance, requires more powerful hardware
ollama pull mixtral
# Good balance of performance and hardware requirements
ollama pull mistral
# Lower resource requirements
ollama pull llama2
```

For multilingual support:
```bash
# Good multilingual capabilities
ollama pull llama3:70b
# Or for lower hardware requirements
ollama pull gemma:7b
```

## 4. Verify Installation
```bash
# Test if Ollama is working
ollama run mistral "Summarize this meeting: Alice and Bob discussed launching a new product in September."
```

## 5. Docker Setup (optional)

If you prefer running Ollama in Docker:

```bash
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

## 6. Configure Ollama API in Your Environment

The Ollama API is accessible at `http://localhost:11434` by default.

Add this to your `.env` file:
```
OLLAMA_API_BASE=http://localhost:11434
OLLAMA_MODEL=mistral
```