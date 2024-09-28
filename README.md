# Break Word Traps

Run server and provide `API_KEY`

```bash
pip install -e .
API_KEY="<API_KEY>" bwt --host "<HOST>" --port "<PORT>"
```

For dev version with reloading use:

```bash
API_KEY="<API_KEY>" uvicorn --host "<HOST>" --port "<PORT>" --reload break_word_traps.endpoints:get_app
```
