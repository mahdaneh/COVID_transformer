import torch
import torch.nn as nn


class InputEmbbeding(nn.Module):
    def __init__(self, image_size, patch_size, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.image_h = image_size[-1]
        self.num_channels = image_size[0]
        self.patch_size = patch_size
        self.input_dim = self.patch_size * self.patch_size * self.num_channels
        self.num_patches = (self.image_h // self.patch_size) ** 2
        self.linear_project = nn.Linear(self.input_dim, self.emb_dim)
        # CLS token initialized as a learnable parameter
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.emb_dim))
        self.pos_embed = nn.Parameter(
            torch.randn(1, 1 + self.num_patches, self.emb_dim)
        )

    def forward(self, x):
        n_batches = x.size(0)
        x = x.view(n_batches, self.num_patches, self.input_dim)
        # linear projection
        x = self.linear_project(x)
        cls_token = self.cls_token.expand(
            n_batches, -1, -1
        )  # Shape: (b, 1, self.emb_dim)
        x = torch.cat((cls_token, x), dim=1)  # (b, np+1, emb_dim)
        x = x + self.pos_embed.expand(n_batches, -1, -1)
        # print('output inputembeding', x.size())
        return x


class HeadAttention(nn.Module):
    def __init__(self, embed_dim, head_dim, dropout=0.1):
        super().__init__()
        self.Q = nn.Linear(embed_dim, head_dim, bias=False)
        self.V = nn.Linear(embed_dim, head_dim, bias=False)
        self.K = nn.Linear(embed_dim, head_dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, z):
        Q_z = self.Q(z)  # z\in[B,D]*w\in[D, D_h] -> [B,D_h]
        V_z = self.V(z)
        K_z = self.K(z)
        A = self.softmax(torch.matmul(Q_z, torch.transpose(K_z, 2, 1)))
        # print('headattention', A.size(), V_z.size())
        output = torch.matmul(A, V_z)
        # print('output attention', output.size())
        # print('final output attention', output.size())
        return output


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, mlp_dim, num_head):
        super().__init__()
        head_dim = embed_dim // num_head
        self.heads = nn.ModuleList(
            [HeadAttention(embed_dim, head_dim, dropout=0.1) for _ in range(num_head)]
        )
        if embed_dim != mlp_dim:
            self.mlp = nn.Sequential(
                nn.Linear(embed_dim, mlp_dim, bias=True),
                nn.GELU(),
                nn.Linear(mlp_dim, embed_dim, bias=True),
            )
        else:
            # experimentally I found using a onelayer mlp is better than a 2 layers mlp.
            self.mlp = nn.Sequential(nn.Linear(embed_dim, embed_dim, bias=True))
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x_norm = self.layer_norm(x)
        # print('1st layerN', x_norm.size())
        x_msa = torch.cat(
            [self.heads[i](x_norm) for i in range(len(self.heads))], dim=2
        )

        x_msa = self.dropout(x_msa) + x
        x_in_mlp = self.layer_norm(x_msa)
        return self.mlp(x_in_mlp) + x_msa


class VIT(nn.Module):
    def __init__(self, config):
        embed_dim = config["embd dim"]
        # experimentally I found using a onelayer mlp is better than a 2 layers mlp.
        if "mlp dim" not in config:
            mlp_dim = config["embd dim"]
        else:
            mlp_dim = config["mlp dim"]
        image_size = config["input size"]
        patch_size = config["patch size"]
        num_layers = config["num layers"]
        num_classes = config["num classes"]
        num_heads = config["num heads"]

        super().__init__()
        assert image_size[1] % patch_size == 0

        if embed_dim % num_heads != 0:
            embed_dim += embed_dim % num_heads

        self.input_embed = InputEmbbeding(
            image_size, patch_size=patch_size, emb_dim=embed_dim
        )
        self.encoders = nn.ModuleList(
            [
                TransformerEncoderBlock(embed_dim, mlp_dim, num_heads)
                for _ in range(num_layers)
            ]
        )
        self.MLP_cls = nn.Sequential(
            nn.Linear(embed_dim, num_classes, bias=True), nn.Softmax(dim=1)
        )

    def forward(self, image):
        x = self.input_embed(image)
        for encoder in self.encoders:
            x = encoder(x)
        z_class = x[:, 0, :].squeeze(1)
        # print('z_class', z_class.size())
        prediction = self.MLP_cls(z_class)
        # print('prediction', prediction.size())
        return prediction


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.cnn_layers = nn.Sequential(
            # First Convolutional Block
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Second Convolutional Block
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Flatten the output for the fully connected layer
            nn.Flatten(),
        )
        # Fully Connected Layers (adjust input features based on your image size and number of conv layers)
        # Example for 32x32 input images after two MaxPool2d layers (each halves spatial dimensions)
        # 32 / 2 / 2 = 8, so 8x8 spatial dimensions
        # 32 channels * 8 * 8 = 2048 input features
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 8 * 8, 128), nn.ReLU(), nn.Linear(128, 3), nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = self.fc_layers(x)
        return x
