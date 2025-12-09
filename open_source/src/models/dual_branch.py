import torch
import torch.nn as nn


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation
        layers = []
        for i in range(2):
            in_ch = in_channels if i == 0 else out_channels
            layers.extend([
                nn.Conv1d(in_ch, out_channels, kernel_size, padding=padding, dilation=dilation),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
        self.net = nn.Sequential(*layers)
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.activation = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = self.residual(x)
        if out.size(-1) != res.size(-1):
            min_len = min(out.size(-1), res.size(-1))
            out = out[..., :min_len]
            res = res[..., :min_len]
        return self.activation(out + res)


class EMGBranch(nn.Module):
    def __init__(self, in_channels, channels, kernel_size, dropout, transformer_heads, transformer_layers, ff_dim):
        super().__init__()
        blocks = []
        for idx, out_channels in enumerate(channels):
            block_in = in_channels if idx == 0 else channels[idx - 1]
            dilation = 2 ** idx
            blocks.append(TemporalBlock(block_in, out_channels, kernel_size, dilation, dropout))
        self.tcn = nn.Sequential(*blocks)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels[-1],
            nhead=transformer_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

    def forward(self, x):
        feats = self.tcn(x)
        seq = feats.permute(2, 0, 1)
        seq = self.transformer(seq)
        return seq.permute(1, 2, 0)


class IMUBranch(nn.Module):
    def __init__(self, in_channels, channels, kernel_size, dropout):
        super().__init__()
        self.enabled = in_channels > 0
        layers = []
        current = in_channels
        for out_channels in channels:
            layers.extend([
                nn.Conv1d(current, out_channels, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            current = out_channels
        self.net = nn.Sequential(*layers) if self.enabled else None
        self.out_channels = current if self.enabled else 0

    def forward(self, x):
        if not self.enabled:
            return x.new_zeros(x.size(0), 0, x.size(2))
        return self.net(x)


class DualBranchModel(nn.Module):
    def __init__(
        self,
        emg_channels: int,
        imu_channels: int,
        num_classes: int,
        gyro_bins: int | None,
        emg_tcn_channels,
        emg_kernel_size,
        emg_dropout,
        transformer_heads,
        transformer_layers,
        transformer_ff_dim,
        imu_cnn_channels,
        imu_kernel_size,
        imu_dropout,
        lstm_hidden,
        lstm_layers,
        attention_mode: str = "emg_to_imu",  # options: none, emg_to_imu, imu_to_emg, bidirectional
        attention_heads: int = 4,
        attention_dropout: float = 0.1,
    ):
        super().__init__()
        self.emg_branch = EMGBranch(
            emg_channels,
            emg_tcn_channels,
            emg_kernel_size,
            emg_dropout,
            transformer_heads,
            transformer_layers,
            transformer_ff_dim,
        )
        self.imu_branch = IMUBranch(imu_channels, imu_cnn_channels, imu_kernel_size, imu_dropout)
        self.attention_mode = attention_mode if self.imu_branch.out_channels > 0 else "none"

        emg_dim = emg_tcn_channels[-1]
        imu_dim = self.imu_branch.out_channels
        if self.attention_mode in {"emg_to_imu", "bidirectional"} and imu_dim > 0:
            self.cross_attn_emg = nn.MultiheadAttention(
                embed_dim=emg_dim,
                num_heads=attention_heads,
                dropout=attention_dropout,
                batch_first=False,
                kdim=imu_dim,
                vdim=imu_dim,
            )
        if self.attention_mode in {"imu_to_emg", "bidirectional"} and imu_dim > 0:
            self.cross_attn_imu = nn.MultiheadAttention(
                embed_dim=imu_dim,
                num_heads=attention_heads,
                dropout=attention_dropout,
                batch_first=False,
                kdim=emg_dim,
                vdim=emg_dim,
            )

        fused_channels = emg_tcn_channels[-1] + self.imu_branch.out_channels
        self.lstm = nn.LSTM(
            input_size=fused_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if lstm_layers > 1 else 0.0,
        )
        self.head_action = nn.Linear(lstm_hidden * 2, num_classes)
        self.head_gyro = nn.Linear(lstm_hidden * 2, gyro_bins) if gyro_bins else None

    def forward(self, emg, imu):
        emg_feats = self.emg_branch(emg)
        imu_feats = self.imu_branch(imu)

        use_attention = self.attention_mode != "none" and imu_feats.size(1) > 0
        if use_attention and self.attention_mode == "emg_to_imu":
            emg_seq = emg_feats.permute(2, 0, 1)
            imu_seq = imu_feats.permute(2, 0, 1)
            attn_emg, _ = self.cross_attn_emg(emg_seq, imu_seq, imu_seq)
            attn_emg = attn_emg.permute(1, 2, 0)
            fused = torch.cat([attn_emg, imu_feats], dim=1)
        elif use_attention and self.attention_mode == "imu_to_emg":
            emg_seq = emg_feats.permute(2, 0, 1)
            imu_seq = imu_feats.permute(2, 0, 1)
            attn_imu, _ = self.cross_attn_imu(imu_seq, emg_seq, emg_seq)
            attn_imu = attn_imu.permute(1, 2, 0)
            fused = torch.cat([emg_feats, attn_imu], dim=1)
        elif use_attention and self.attention_mode == "bidirectional":
            emg_seq = emg_feats.permute(2, 0, 1)
            imu_seq = imu_feats.permute(2, 0, 1)
            attn_emg, _ = self.cross_attn_emg(emg_seq, imu_seq, imu_seq)
            attn_imu, _ = self.cross_attn_imu(imu_seq, emg_seq, emg_seq)
            fused = torch.cat([attn_emg.permute(1, 2, 0), attn_imu.permute(1, 2, 0)], dim=1)
        else:
            fused = torch.cat([emg_feats, imu_feats], dim=1) if imu_feats.size(1) > 0 else emg_feats

        lstm_out, _ = self.lstm(fused.permute(0, 2, 1))
        feats = lstm_out[:, -1, :]
        action_logits = self.head_action(feats)
        gyro_logits = self.head_gyro(feats) if self.head_gyro is not None else None
        return action_logits, gyro_logits
