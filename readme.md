# CLIP Model file and pretrained state_dict

An replication of [CLIP](https://github.com/openai/CLIP). Since the offical repo only provides **a JIT version model (VIT-B-32)**, it's not convenient to modify. And here is an **non-JIT version model**. By this, you can change as you will.

## Usage

### Install

Please ref to the official repo [CLIP](https://github.com/openai/CLIP), and check the [notebook](https://github.com/openai/CLIP/blob/main/Interacting_with_CLIP.ipynb). The ${BPEPATH} in below is [here](https://github.com/openai/CLIP/blob/main/bpe_simple_vocab_16e6.txt.gz) and ${MODELPATH} is the state_dict extract from the JIT version model, and provided [here](https://drive.google.com/file/d/1BVYRs2QB-Va79oAqpW5fcrw-TEmMj4Kh/view?usp=sharing)

### Inference

An example to use it for inference is here: [**Test on text branch only**]
```python
from CLIP import CLIP
from clip_tokenizer import SimpleTokenizer 
import torch

tokenizer = SimpleTokenizer(bpe_path=${BPEPATH})
model = CLIP(attention_probs_dropout_prob=0, hidden_dropout_prob=0)
state_dict = torch.load(${MODELPATH})
model.load_state_dict(state_dict)
is_fp16 = False
device = "cuda" if torch.cuda.is_available() else "cpu"
if is_fp16:
    model.to(device=device).eval().half()
else:
    model.to(device=device).eval().float()

with torch.no_grad():
    query = ["What will be Covid-19's long-term global economic impact?"]
    text_tokens = [tokenizer.encode("This is " + desc + "<|endoftext|>") for desc in query]
    text_input = torch.zeros(len(text_tokens), model.context_length, dtype=torch.long)

    for i, tokens in enumerate(text_tokens):
        # truncate and keep EOS unremoved
        length = min(len(tokens), model.context_length.item())
        tokens = torch.tensor(tokens[:length-1]+tokens[-1:])
        text_input[i, :length] = tokens

    emb = model.encode_text(text_input.to(device=device))
```

## Thanks

Thanks to excellent work by [CLIP](https://github.com/openai/CLIP)