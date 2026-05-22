import torch

from src.data import load_corpus, Tokenizer
from src.model import GPT


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    ckpt = torch.load('checkpoints/model.pt', map_location=device)

    model_cfg_dict = ckpt['config']['model']
    model = GPT(**model_cfg_dict)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    tok = Tokenizer(load_corpus())
    prompt = "ROMEO:\n"                                                                               
    ids = tok.encode_to_tensor(prompt).unsqueeze(0).to(device)   # (1, L) 

    out_ids = model.generate(ids, max_new_tokens=500, temperature=0.7, top_k=10)
    text = tok.decode(out_ids[0].tolist())
    print(text)


if __name__ == '__main__':
    main()
