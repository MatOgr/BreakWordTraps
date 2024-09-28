# Break Word Traps

Install dependencies

```bash
pip install -e .
```

Run server and provide `API_KEY`

```bash
API_KEY="<API_KEY>" bwt run_backend "<ASR_SERVICE_ADDRESS>" "<FRE_SERVICE_ADDRESS>" --host "<HOST>" --port "<PORT>"
```

## Run ASR service

Install dependencies

```bash
pip install -e '.[asr]'
```

```bash
API_KEY="<API_KEY>" bwt run_subservice --host "<HOST>" --port "<PORT>" --llm-server-address "<LLM_SERVICE_ADDRESS>"
```
