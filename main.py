import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
import warnings
import random
import os
import time
import math

warnings.filterwarnings('ignore')

# ==================== 随机种子 ====================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, x):
        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0) + 1e-8

    def transform(self, x):
        return (x - self.mean) / self.std

class GatedConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.conv_feat = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.conv_gate = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        feat = self.conv_feat(x)
        gate = self.sigmoid(self.conv_gate(x))
        return feat * gate

class ResidualBlock(nn.Module):
    """残差块，增强特征学习"""
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.net(x))


class PositionalEncoding1D(nn.Module):
    """1D位置编码，用于SNP序列"""

    def __init__(self, d_model, max_len=20000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class SNPEncoder(nn.Module):

    def __init__(self, snp_len, d_model=128, nhead=8, num_layers=3, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.embed_conv = nn.Sequential(
            GatedConv1d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(16),
            nn.GELU(),
            GatedConv1d(16, d_model, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        # 卷积输出长度
        self.conv_out_len = ((snp_len + 2 * 3 - 7) // 2 + 1 + 2 * 2 - 5) // 2 + 1

        self.pos_encoder = PositionalEncoding1D(d_model, max_len=self.conv_out_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, snp):
        x = snp.unsqueeze(1)
        x = self.embed_conv(x)
        x = x.transpose(1, 2)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = self.layer_norm(x)
        return x

class EnvironmentEncoder(nn.Module):

    def __init__(self, env_dim, d_model=128, dropout=0.1):
        super().__init__()
        self.fc = nn.Linear(1, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.res = ResidualBlock(d_model)

    def forward(self, env):
        x = env.unsqueeze(-1)
        x = self.fc(x)
        x = self.act(self.norm(x))
        x = self.dropout(x)
        return x

class BidirectionalCrossAttention(nn.Module):


    def __init__(self, d_model, nhead=8, dropout=0.1):
        super().__init__()

        self.env2gene_mha = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.gene2env_mha = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.norm_gene = nn.LayerNorm(d_model)
        self.norm_env = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, gene_seq, env_seq):

        env_out, _ = self.env2gene_mha(query=env_seq, key=gene_seq, value=gene_seq)   # (B, M, d)
        env_out = self.norm_env(env_seq + self.dropout(env_out))

        gene_out, _ = self.gene2env_mha(query=gene_seq, key=env_seq, value=env_seq)   # (B, L, d)
        gene_out = self.norm_gene(gene_seq + self.dropout(gene_out))

        return gene_out, env_out

class CrossAttentionNet(nn.Module):
    def __init__(self, snp_len, env_dim, d_model=128, nhead=8,
                 snp_transformer_layers=1, snp_ffn_dim=512,
                 fusion_dims=[512, 256, 64], dropout=0.15):
        super().__init__()
        self.snp_encoder = SNPEncoder(
            snp_len=snp_len, d_model=d_model, nhead=nhead,
            num_layers=snp_transformer_layers, dim_feedforward=snp_ffn_dim,
            dropout=dropout
        )
        self.env_encoder = EnvironmentEncoder(
            env_dim=env_dim, d_model=d_model, dropout=dropout
        )
        self.bi_cross_attn = BidirectionalCrossAttention(d_model, nhead, dropout)

        fusion_dim = d_model * 2
        self.fusion_res = ResidualBlock(fusion_dim)
        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dims[0]),
            nn.LayerNorm(fusion_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dims[0], 1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, snp, env):
        gene_seq = self.snp_encoder(snp)
        env_seq = self.env_encoder(env)

        gene_out, env_out = self.bi_cross_attn(gene_seq, env_seq)


        gene_vec = gene_out.mean(dim=1)
        env_vec = env_out.mean(dim=1)

        fusion = torch.cat([gene_vec, env_vec], dim=-1)
        fusion = self.fusion_res(fusion)
        out = self.regressor(fusion).squeeze(-1)
        return out

class RFNetWrapper(nn.Module):
    def __init__(self, snp_len, env_dim, geno_tensor, **kwargs):
        super().__init__()
        self.register_buffer('geno_tensor', geno_tensor)
        self.canet = CrossAttentionNet(snp_len, env_dim, **kwargs)

    def forward(self, gene_idx, env, training=True):
        snp = self.geno_tensor[gene_idx]
        if training:
            snp = snp + torch.randn_like(snp) * 0.01
            env = env + torch.randn_like(env) * 0.01
        return self.canet(snp, env)

# ==================== 数据集 ====================
class RFNetDataset(Dataset):
    def __init__(self, df, env_dict, env_scaler, geno_name2idx, device):
        self.df = df.reset_index(drop=True)
        self.env_dict = env_dict
        self.env_scaler = env_scaler
        self.geno_name2idx = geno_name2idx
        self.device = device

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        env_name = str(row.Environment).strip()
        hybrid = str(row.Hybrid).strip()
        gene_idx = self.geno_name2idx[hybrid]
        y = torch.tensor(row.Yield, dtype=torch.float32, device=self.device)
        env_raw = self.env_dict[env_name]
        env_norm = self.env_scaler.transform(env_raw.cpu().numpy())
        env_norm = torch.tensor(env_norm, dtype=torch.float32, device=self.device).squeeze(0)
        return env_norm, gene_idx, y

# ==================== 数据加载函数 ====================
def load_geno_tensor(path, device):
    df = pd.read_csv(path, index_col=0).round(5)
    geno_np = df.values
    geno_tensor = torch.tensor(geno_np, dtype=torch.float32, device=device)
    name2idx = {str(idx).strip(): i for i, idx in enumerate(df.index)}

    return name2idx, geno_tensor, geno_tensor.shape[1]

def load_pheno(path):
    df = pd.read_csv(path)
    pheno = df.iloc[:, [0, 1, 2]].copy()
    pheno.columns = ['Environment', 'Hybrid', 'Yield']
    pheno = pheno.dropna().round(5)
    return pheno

def load_env(path, device):
    df = pd.read_csv(path).round(5)
    env_scaler = StandardScaler()
    env_scaler.fit(df.iloc[:, 1:].values)
    env_dict = {}
    for _, row in df.iterrows():
        name = str(row.iloc[0]).strip()
        vals = row.iloc[1:].astype(float).fillna(0).values
        env_dict[name] = torch.tensor(vals.reshape(1, -1), dtype=torch.float32, device=device)

    return env_dict, env_scaler, len(df.columns) - 1

def filter_matched(pheno_df, geno_name2idx, env_dict):
    keep = []
    for i, (env, hybrid, _) in enumerate(pheno_df.itertuples(index=False)):
        if hybrid in geno_name2idx and env in env_dict:
            keep.append(i)
    res = pheno_df.iloc[keep].reset_index(drop=True)
    return res

# ==================== 表型标准化 ====================
class PhenoNorm:
    def fit(self, df):
        self.m = df['Yield'].mean()
        self.s = df['Yield'].std() or 1.0

    def norm(self, v): return (v - self.m) / self.s
    def denorm(self, v): return v * self.s + self.m

# ==================== 评估函数 ====================
def eval_safe(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for env, gene_idx, y in loader:
            pred = model(gene_idx, env, training=False).cpu().numpy().ravel()
            true = y.cpu().numpy().ravel()
            preds.extend(pred)
            trues.extend(true)
    p = np.array(preds)
    t = np.array(trues)
    if len(p) < 2:
        return 0.0, 0.0, 0.0, 0.0, p, t
    r, _ = pearsonr(t, p)
    r = np.clip(r, -1, 1)
    r2 = r2_score(t, p)
    mse = mean_squared_error(t, p)
    rmse = np.sqrt(mse)
    return r, r2, mse, rmse, p, t

# ==================== 早停类 ====================
class EarlyStopping:
    def __init__(self, patience=50, min_delta=1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = -np.inf
        self.best_state = None
        self.best_epoch = 0

    def step(self, score, model, epoch):
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.best_epoch = epoch
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

# ==================== 训练函数 ====================
class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, true):
        return self.mse(pred, true)

def train(model, train_loader, test_loader, epochs, lr, device, patience=50):
    criterion = Loss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.00001)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2, eta_min=1e-6)
    early_stop = EarlyStopping(patience=patience)

    for epoch in range(epochs):
        model.train()
        loss_sum, cnt = 0.0, 0
        for env, gene_idx, y in train_loader:
            optimizer.zero_grad()
            pred = model(gene_idx, env, training=True)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.8)
            optimizer.step()
            loss_sum += loss.item()
            cnt += 1

        scheduler.step()
        avg_loss = loss_sum / cnt

        test_r, _, _, _, _, _ = eval_safe(model, test_loader, device)
        if early_stop.step(test_r, model, epoch):
            print(f"🛑 早停 @ epoch {epoch + 1}, 最佳测试PCC={early_stop.best_score:.4f}")
            break

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:3d} | Loss {avg_loss:.4f} | test PCC {test_r:.4f} | Best {early_stop.best_score:.4f}")

    if early_stop.best_state is not None:
        model.load_state_dict(early_stop.best_state)
        best_test_pcc = early_stop.best_score
    else:
        best_test_pcc = test_r
    return model, best_test_pcc

# ==================== 十折交叉验证 ====================
def kfold_cv(pheno_df, env_dict, env_scaler, geno_name2idx, geno_tensor, cfg, device, k=10):
    save_root = r"D:/dest/exgepdata/albation-pe"
    os.makedirs(save_root, exist_ok=True)
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    res, all_preds = [], []
    total_start = time.time()

    for fold, (train_idx, test_idx) in enumerate(kf.split(pheno_df), 1):
        fold_start = time.time()

        print(f"\n========== 第 {fold}/{k} 折 ==========")

        train_df = pheno_df.iloc[train_idx]
        test_df = pheno_df.iloc[test_idx]

        norm = PhenoNorm()
        norm.fit(train_df)
        train_df = train_df.copy()
        test_df = test_df.copy()
        train_df['Yield'] = norm.norm(train_df['Yield'])
        test_df['Yield'] = norm.norm(test_df['Yield'])

        train_loader = DataLoader(RFNetDataset(train_df, env_dict, env_scaler, geno_name2idx, device),
                                  batch_size=cfg['bs'], shuffle=True, num_workers=0)
        test_loader = DataLoader(RFNetDataset(test_df, env_dict, env_scaler, geno_name2idx, device),
                                 batch_size=cfg['bs'], shuffle=False, num_workers=0)

        model = RFNetWrapper(
            snp_len=cfg['geno_dim'],
            env_dim=cfg['env_in'],
            geno_tensor=geno_tensor,
            d_model=cfg['d_model'],
            nhead=cfg['nhead'],
            snp_transformer_layers=cfg['snp_transformer_layers'],
            snp_ffn_dim=cfg['snp_ffn_dim'],
            fusion_dims=cfg['fusion_dims'],
            dropout=cfg['dropout']
        ).to(device)

        model, best_test_r = train(model, train_loader, test_loader, cfg['epochs'], cfg['lr'], device, cfg['patience'])

        torch.save(model.state_dict(), os.path.join(save_root, f"best_model_fold{fold}.pth"))
        print(f"💾 最优模型已保存至: best_model_fold{fold}.pth")

        r, r2, mse, rmse, p, t = eval_safe(model, test_loader, device)
        p = norm.denorm(p)
        t = norm.denorm(t)

        fold_time = time.time() - fold_start
        print(f"✅ 第 {fold} 折 | 测试集 PCC={r:.4f} | R2={r2:.4f} | RMSE={rmse:.4f} | 耗时 {fold_time:.2f}s")
        res.append({
            "Fold": fold,
            "test_pearson": round(r, 4),
            "test_r2": round(r2, 4),
            "test_mse": round(mse, 4),
            "test_rmse": round(rmse, 4),
            "best_test_pearson": round(best_test_r, 4),
            "fold_time_sec": round(fold_time, 2),
        })

        pred_df = test_df.copy()
        pred_df['True_Yield'] = np.round(t, 4)
        pred_df['Pred_Yield'] = np.round(p, 4)
        pred_df['Fold'] = fold
        all_preds.append(pred_df)

    print(f"\n总耗时: {time.time() - total_start:.2f}s")
    pd.DataFrame(res).to_csv(os.path.join(save_root, "kfold_results_model1.csv"), index=False)
    pd.concat(all_preds, ignore_index=True).to_csv(os.path.join(save_root, "kfold_predictions_model1.csv"), index=False)
    print("✅ 序列化双向交叉注意力模型结果已保存至:", save_root)
    return res

# ==================== 主函数 ====================
def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("使用设备:", device)

    paths = {
        "geno": "D:/ayc/data/EXGEP_22(处理)/genotype.csv",
        "pheno": "D:/ayc/data/cleaned_phenotype1.csv",
        "env": "D:/ayc/data/merged.csv"
    }

    cfg = {
        "bs": 32,
        "epochs": 200,
        "lr": 0.0005,
        "patience": 50,
        "d_model": 64,
        "nhead": 8,
        "snp_transformer_layers": 1,
        "snp_ffn_dim": 128,
        "fusion_dims": [64],
        "dropout": 0.1
    }

    geno_name2idx, geno_tensor, snp_len = load_geno_tensor(paths["geno"], device)
    pheno_df = load_pheno(paths["pheno"])
    env_dict, env_scaler, env_dim = load_env(paths["env"], device)
    pheno_df = filter_matched(pheno_df, geno_name2idx, env_dict)

    cfg['geno_dim'] = snp_len
    cfg['env_in'] = env_dim

    res = kfold_cv(pheno_df, env_dict, env_scaler, geno_name2idx, geno_tensor, cfg, device, k=10)
    df = pd.DataFrame(res)

    print("\n==================== 十折交叉验证汇总 ====================")
    print(f"平均 PCC:   {df['test_pearson'].mean():.4f} ± {df['test_pearson'].std():.4f}")
    print(f"平均 R2:    {df['test_r2'].mean():.4f} ± {df['test_r2'].std():.4f}")
    print(f"平均 RMSE:  {df['test_rmse'].mean():.4f} ± {df['test_rmse'].std():.4f}")
    print(f"平均每折耗时: {df['fold_time_sec'].mean():.2f} ± {df['fold_time_sec'].std():.2f} 秒")

if __name__ == "__main__":
    main()
