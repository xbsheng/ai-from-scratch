import torch
from model import Encoder, LayerNormal


def test_encoder():
    torch.manual_seed(0)
    batch, seq_len, d_model = 2, 8, 32
    enc = Encoder(num_layers=3, d_model=d_model, num_heads=4, dropout=0.1, d_ff=64)

    x = torch.randn(batch, seq_len, d_model)
    out = enc(x)

    assert out.shape == (batch, seq_len, d_model), out.shape

    # 不变性: 相同输入行在 dropout=0 时输出应一致
    torch.manual_seed(0)
    enc0 = Encoder(num_layers=3, d_model=d_model, num_heads=4, dropout=0.0, d_ff=64)
    x_same = torch.randn(batch, 1, d_model).repeat(1, seq_len, 1)
    out_same = enc0(x_same)
    torch.testing.assert_close(out_same[:, 0, :], out_same[:, 1, :])

    # mask 可选: 传 mask 与不传 mask 形状一致
    mask = torch.ones(batch, 1, 1, seq_len)
    assert enc(x, mask).shape == out.shape


def test_encoder_layer_norm_is_last():
    enc = Encoder(num_layers=1, d_model=16, num_heads=2, dropout=0.0, d_ff=32)
    assert isinstance(enc.normal, LayerNormal)


if __name__ == "__main__":
    test_encoder()
    test_encoder_layer_norm_is_last()
    print("✅ all tests passed")
