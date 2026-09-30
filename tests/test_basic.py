import torch
from model import GPT

def test_forward_shape():
    model = GPT(vocab_size=65, embed_dim=32, num_heads=2, num_layers=2, block_size=16)
    x = torch.randint(0, 65, (2, 16))
    logits, loss = model(x, x)
    assert logits.shape == (2, 16, 65)
    assert loss.item() > 0

def test_generate_extends_sequence():
    model = GPT(vocab_size=65, embed_dim=32, num_heads=2, num_layers=2, block_size=16)
    x = torch.zeros((1, 1), dtype=torch.long)
    out = model.generate(x, max_new_tokens=10)
    assert out.shape == (1, 11)

def test_causal_mask_blocks_future():
    model = GPT(vocab_size=10, embed_dim=16, num_heads=2, num_layers=1, block_size=8)
    head = model.blocks[0].attn.heads[0]
    assert head.mask.shape == (8, 8)
    assert head.mask[0, 1] == 0  # position 0 cannot see position 1 (the future)