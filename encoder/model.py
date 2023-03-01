from torch import nn

class EncDec(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # Output size: (batch_size, 16, 25, 25)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # Output size: (batch_size, 32, 13, 13)
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1), # Output size: (batch_size, 16, 7, 7)
            nn.ReLU(),
        )
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 32, kernel_size=3, stride=2, padding=1), # Output size: (batch_size, 32, 13, 13)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1), # Output size: (batch_size, 16, 25, 25)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2, padding=0, output_padding=0)  # Output size: (batch_size, 1, 50, 50)
            ,nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x