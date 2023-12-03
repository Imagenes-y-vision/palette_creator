import torch
from torch import nn, optim
from torch.nn import functional as F
from palette_creator.methods.abstract_method import Method
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import numpy as np
from palette_creator.utils import quantize_image


class NeuralNetwork(Method):
    PATH_PRETRAINED_MODEL = "data/nga_neural_network.pth"
    def __init__(self, palette_size: int = 6, use_pretrained: bool = True, train_epochs: int = 0, train_images: np.ndarray = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PaletteNN(image_dimension=(3, 512, 512), palette_size=palette_size).to(self.device)
        self.criterion = PaletteLoss(alpha=0)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        if use_pretrained:
            self.model.load_state_dict(torch.load(self.PATH_PRETRAINED_MODEL))
        if train_epochs:
            dataset = CustomDataset(train_images)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
            self.__train(train_epochs, dataloader)
        
        self.model.eval()
        
    def create_palette(self, image):
        if image.shape != (512, 512, 3):
            raise Exception("La imagen debe estar en formato 512x512x3")
        input_ = torch.from_numpy(image).permute(2, 0, 1).float().to(self.device) 
        with torch.no_grad():
            output = self.model(input_).cpu()
        palette = output.reshape(6,3).numpy()
        quantized_image = quantize_image(image, palette).reshape(-1, 3)
        proportions = np.unique(quantized_image, return_counts=True, axis=0)[1] / len(quantized_image)
        return (palette*255).astype(int), proportions
        
    def __train(self, num_epochs, data_loader):
        self.model.train()
        train_loss = 0
        for epoch in range(num_epochs):
            with tqdm(
                    total=len(data_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch"
            ) as pbar:
                for batch_images in data_loader:
                    batch_images = batch_images.to(self.device)
                    self.optimizer.zero_grad()
                    
                    palettes_features = self.model(batch_images)
                    palettes = palettes_features.reshape(-1, 6, 3)
                    # print(palettes.device)
                    # Actualizar la barra de progreso
                    loss = self.criterion(palettes, batch_images)
                    # import pdb; pdb.set_trace()
                    loss.backward()
                    # import pdb; pdb.set_trace()
                    train_loss += loss.item()
                    self.optimizer.step()
                    pbar.set_postfix({"loss": loss.item()})
                    pbar.update(1)
                    
                    # print(
                    #     "====> Epoch: {} Average loss: {:.4f}".format(
                    #         epoch + 1, train_loss / len(data_loader.dataset)
                    #     )
                    #)

    
class PaletteNN(nn.Module):
    def __init__(self, image_dimension: tuple, palette_size=6):
        super(PaletteNN, self).__init__()
        n_convolutions = 4
        self.dim_output_conv = image_dimension[1] // 2**n_convolutions
        input_channels = image_dimension[0]
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * self.dim_output_conv**2, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.palette = nn.Linear(
            512, palette_size * 3
        )  # palette_size * n_channels (e.g., a palette of 6 colors rgb, then 18 neurons)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 32 * self.dim_output_conv**2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        output = torch.sigmoid(self.palette(x))
        return output

class PaletteLoss(nn.Module):
    """
    La función de pérdida está dada por 2 elementos principales:
    El primer elemento es el MSE de la imagen cuantizada por una paleta y la imagen real.
    El segundo elemento es la pérdida de la paleta, que entre más cercano a 1 indica que los colores de
    la paleta son muy similares. Y entre más cercano a 0 indica que los colores de la paleta son distantes.
    Por lo tanto, la función de pérdida queda así:
                f(paleta, imagen) = (1-alpha)*MSE(imagen_cuantizada, imagen) - alpha*PALETTE_LOSS(paleta)
    donde alpha es la sensibilidad de la función de pérdida de la paleta en  la pérdida general.
    A partir de experimentación, encontramos un alpha estable en 0.001
    """

    def __init__(self, alpha=0.001):
        super(PaletteLoss, self).__init__()
        self.alpha = alpha

    def forward(self, palettes, images):
        # 1. Obtener el MSE de la imagen cuantizada por la paleta y la imagen real
        quantized_images = self.quantize_images(images, palettes)
        mse_loss = F.mse_loss(quantized_images, images)

        # 2. Obtener la pérdida de la paleta
        # palette_loss = self.palette_loss(palettes)
        # Este lo definimos como la distancia entre cada uno de los colores de la paleta
        # Si los colores son muy similares, no disminuye el error. Si los colores son distantes, significa
        # que la paleta está mejor distribuida. Lo cual disminuirá el error.

        # calculate distance in palette
        batch_size, n_colors, n_channels = palettes.shape
        # palettes1 = palettes.reshape(batch_size, n_colors, 1, n_channels)
        # palettes2 = palettes.reshape(batch_size, 1, n_colors, n_channels)
        mask = (
            torch.triu(torch.ones(batch_size, n_colors, n_colors), diagonal=1)
            .reshape(batch_size, n_colors, n_colors, 1)
            .to(torch.device("cuda"))
        )
        total_combinations = n_colors * (n_colors - 1) / 2
        difference = palettes[:, :, None, :] - palettes[:, None, :, :]
        palette_loss = torch.sum(torch.norm(((difference) * mask), dim=3)) / (
            total_combinations * batch_size
        )

        return mse_loss - self.alpha * palette_loss

    def palette_loss(self, palettes):
        batch_size, n_colors, n_channels = palettes.shape
        mask = (
            torch.triu(torch.ones(batch_size, n_colors, n_colors), diagonal=1)
            .reshape(batch_size, n_colors, n_colors, 1)
            .to(torch.device("cuda"))
        )
        total_combinations = n_colors * (n_colors - 1) / 2
        difference = palettes[:, :, None, :] - palettes[:, None, :, :]
        mean_distance_palettes = torch.sum(torch.norm(((difference) * mask), dim=3)) / (
            total_combinations * batch_size
        )
        return 1 - mean_distance_palettes

    def quantize_images(
        self, images: torch.Tensor, palettes: torch.Tensor
    ) -> torch.Tensor:
        batch_size, C, H, W = images.shape
        num_colors = palettes.shape[1]

        # Cambiar el orden de las dimensiones de las imágenes a (batch_size, H, W, C)
        images = images.permute(0, 2, 3, 1).contiguous()

        # Redimensionar las imágenes y las paletas para el cálculo de la distancia
        reshaped_images = images.view(
            batch_size, -1, 1, 3
        )  # (batch_size, num_pixels, 1, 3)
        reshaped_palettes = palettes.view(
            batch_size, 1, num_colors, 3
        )  # (batch_size, 1, num_colors, 3)

        # Calcular la distancia L2 (Euclidean) entre cada píxel y cada color de la paleta
        distances = torch.norm(reshaped_images - reshaped_palettes, dim=3)

        # Encontrar el índice del color más cercano para cada píxel
        nearest_color_indices = torch.argmin(distances, dim=2)

        # Mapear los píxeles a los colores más cercanos
        mapped_images = torch.gather(
            palettes, 1, nearest_color_indices.unsqueeze(2).expand(-1, -1, 3)
        )
        mapped_images = mapped_images.view(batch_size, H, W, C)

        # Cambiar el orden de las dimensiones de vuelta a (batch_size, C, H, W)
        return mapped_images.permute(0, 3, 1, 2)

class CustomDataset(Dataset):
    def __init__(self, numpy_array, transforms=None):
        self.images = torch.from_numpy(numpy_array).float()
        self.images = self.images.permute(0, 3, 1, 2)
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]

        if self.transforms:
            image = self.transforms(image)

        return image

