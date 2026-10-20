"""
Quantune Chat Playground Notebook Cells
======================================
Builds the final cells of the generated Colab notebook: a Gradio chat
playground that serves the freshly fine-tuned model behind a public share
link, so it can be tested straight from Quantune's "Test Your Model" page
(Step 5 → Colab playground) without downloading or converting anything.

Kept as its own module so the playground cells evolve independently of the
core training notebook template in colab_generator.py; they are composed by
quantune/training/__init__.py.
"""

import json
from typing import Any, Dict, List, Optional

# Marker used to keep the cells from being appended twice
PLAYGROUND_MARKER = "quantune-chat-playground"
LEGACY_PLAYGROUND_MARKER = "tunekit-chat-playground"


def extract_system_prompt(dataset_jsonl: str) -> Optional[str]:
    """Return the system prompt the dataset trains with, if it has one."""
    try:
        first_line = (dataset_jsonl or "").strip().split("\n")[0]
        messages = json.loads(first_line).get("messages", [])
        if messages and messages[0].get("role") == "system":
            return messages[0].get("content", "") or None
    except (json.JSONDecodeError, IndexError, KeyError, AttributeError):
        pass
    return None


def _markdown_cell(source: str) -> Dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def _code_cell(source: str, collapsed: bool = False) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    if collapsed:
        metadata["cellView"] = "form"
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": metadata,
        "outputs": [],
        "source": source,
    }


def build_playground_cells(model_name: str, system_prompt: Optional[str]) -> List[Dict[str, Any]]:
    """The two notebook cells (markdown + code) for the chat playground."""
    prompt_literal = json.dumps(system_prompt or "")

    markdown = f"""## Optional: Launch a Chat Playground

Run the cell below and it starts a small chat UI **inside this Colab session** and
prints a public link:

```
Running on public URL: https://xxxxxxxx.gradio.live
```

Paste that link into **Quantune → Test Your Model → Colab playground** to chat with
the model you just trained. Nothing to download, nothing to install locally.

> The cell keeps running while the playground is live - that *is* the server. If you
> clicked **Run all**, this is also why the run does not appear to finish; stop the
> cell when you are done testing.
"""

    code = f'''#@title Launch a chat playground (public share link) [click to expand]
#@markdown Runs your fine-tuned model behind a Gradio chat UI and prints a link to
#@markdown paste into Quantune > Test Your Model > Colab playground.
launch_playground = True  #@param {{type:"boolean"}}

# {PLAYGROUND_MARKER}
if launch_playground:
    try:
        import gradio as gr
    except ImportError:
        import subprocess, sys
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gradio"], check=True)
        import gradio as gr

    FastLanguageModel.for_inference(model)

    SYSTEM_PROMPT = {prompt_literal}

    def _turns_to_messages(history):
        """Gradio hands over either [user, assistant] pairs or role/content dicts."""
        messages = []
        for turn in history or []:
            if isinstance(turn, dict):
                role = turn.get("role")
                if role in ("user", "assistant") and turn.get("content"):
                    messages.append({{"role": role, "content": turn["content"]}})
            elif isinstance(turn, (list, tuple)) and len(turn) == 2:
                if turn[0]:
                    messages.append({{"role": "user", "content": turn[0]}})
                if turn[1]:
                    messages.append({{"role": "assistant", "content": turn[1]}})
        return messages

    def chat_with_model(message, history):
        messages = []
        if SYSTEM_PROMPT:
            messages.append({{"role": "system", "content": SYSTEM_PROMPT}})
        messages.extend(_turns_to_messages(history))
        messages.append({{"role": "user", "content": message}})

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            use_cache=True,
        )
        return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    demo = gr.ChatInterface(
        fn=chat_with_model,
        title="Quantune playground - {model_name}",
        description="Chat with your fine-tuned model, then copy the public URL printed below into Quantune > Test Your Model > Colab playground.",
    )
    demo.launch(share=True, debug=False)
'''

    return [_markdown_cell(markdown), _code_cell(code, collapsed=True)]


def append_playground_cells(notebook_json: str, model_name: str, dataset_jsonl: str = "") -> str:
    """
    Append the chat playground cells to a notebook generated by
    colab_generator.generate_training_notebook (idempotent).
    """
    notebook = json.loads(notebook_json)
    cells = notebook.setdefault("cells", [])

    if any(PLAYGROUND_MARKER in json.dumps(cell.get("source", "")) or LEGACY_PLAYGROUND_MARKER in json.dumps(cell.get("source", "")) for cell in cells):
        return notebook_json

    cells.extend(build_playground_cells(model_name, extract_system_prompt(dataset_jsonl)))
    return json.dumps(notebook, indent=2)
