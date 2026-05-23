import argparse
from pytest import Parser
import torch

from src.data import load_corpus, Tokenizer
from src.model import GPT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate text from trained GPT')
    parser.add_argument('--prompt', type=str, default='\n')
    parser.add_argument('--max-new-tokens', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top-k', type=int, default=None)
    parser.add_argument('--top-p', type=float, default=None)
    parser.add_argument('--use-cache', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=None)
    return parser.parse_args()


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    args = parse_args()
    prompt = args.prompt.encode().decode('unicode_escape')

    ckpt = torch.load('checkpoints/model.pt', map_location=device)

    torch.manual_seed(args.seed)

    model_cfg_dict = ckpt['config']['model']
    model = GPT(**model_cfg_dict)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    tok = Tokenizer(load_corpus())                                                                             
    ids = tok.encode_to_tensor(prompt).unsqueeze(0).to(device)   # (1, L) 

    out_ids = model.generate(
        ids, 
        max_new_tokens=args.max_new_tokens, 
        temperature=args.temperature, 
        top_k=args.top_k,
        top_p=args.top_p,
        use_cache=args.use_cache
        )
    text = tok.decode(out_ids[0].tolist())
    print(text)


if __name__ == '__main__':
    main()
