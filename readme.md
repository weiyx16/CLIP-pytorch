# CLIP Model file and pretrained state_dict

An replication of [CLIP](https://github.com/openai/CLIP). Since the offical repo only provides **a JIT version model (VIT-B-32)**, it's not convenient to modify. And here is an **non-JIT version model**. By this, you can change as you will.

## Usage

### Install

Please ref to the official repo [CLIP](https://github.com/openai/CLIP), and check the [notebook](https://github.com/openai/CLIP/blob/main/Interacting_with_CLIP.ipynb). The ${BPEPATH} in below is [here](https://github.com/openai/CLIP/blob/main/bpe_simple_vocab_16e6.txt.gz) and ${MODELPATH} is the state_dict extract from the JIT version model, and provided [here](https://drive.google.com/file/d/1BVYRs2QB-Va79oAqpW5fcrw-TEmMj4Kh/view?usp=sharing)

### Inference

An example to use it for inference is here:
```python
from CLIP import CLIP, build_transform
from clip_tokenizer import SimpleTokenizer 
from PIL import Image
import torch

model = CLIP(attention_probs_dropout_prob=0, hidden_dropout_prob=0)
model.load_state_dict(state_dict = torch.load(${MODELPATH}))
tokenizer = SimpleTokenizer(bpe_path=${BPEPATH}, context_length=model.context_length.item())
transform = build_transform(model.input_resolution.item())
is_fp16 = False
device = "cuda" if torch.cuda.is_available() else "cpu"
if is_fp16:
    model.to(device=device).eval().half()
else:
    model.to(device=device).eval().float()

with torch.no_grad():
    query = ["a diagram", "a dog", "a cat"]
    text = tokenizer.encode(query).to(device)
    text_features = model.encode_text(text)

    image = transform(Image.open("./CLIP.png")).unsqueeze(0).to(device)
    image_features = model.encode_image(image)

    logits_per_image, logits_per_text = model(image, text, return_loss=False)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()
print("Label probs:", probs) # prints: [[0.99558276 0.00217687 0.00224036]]
```

## Thanks

Thanks to excellent work by [CLIP](https://github.com/openai/CLIP)