"""
VAE (Variational Autoencoder) 실습
확률론적 생성 모델의 구현과 ELBO 분석

이 실습에서는 다음을 다룹니다:
1. VAE 기본 아키텍처 구현
2. Reparameterization Trick 이해
3. ELBO 손실 함수 분해
4. 잠재 공간 시각화
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE

# 1. VAE 아키텍처 정의
class VAE(nn.Module):
    def __init__(self, input_size=784, hidden_size=400, latent_size=20):
        super(VAE, self).__init__()
        
        # 인코더 (q_φ(z|x))
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # 잠재 변수의 평균과 분산 출력
        self.fc_mu = nn.Linear(hidden_size, latent_size)      # μ(x)
        self.fc_logvar = nn.Linear(hidden_size, latent_size)  # log σ²(x)
        
        # 디코더 (p_θ(x|z))
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()  # 픽셀 값을 [0,1] 범위로 제한
        )
        
        self.latent_size = latent_size
    
    def encode(self, x):
        """
        인코더: x → (μ, log σ²)
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization Trick: ε ~ N(0,I), z = μ + σ⊙ε
        이를 통해 확률적 노드를 미분 가능하게 만듦
        """
        if self.training:
            # 표준편차 계산: σ = exp(log σ² / 2)
            std = torch.exp(0.5 * logvar)
            # 표준 정규분포에서 샘플링
            eps = torch.randn_like(std)
            # z = μ + σ⊙ε
            return mu + eps * std
        else:
            # 추론 시에는 평균값 사용
            return mu
    
    def decode(self, z):
        """
        디코더: z → x_reconstructed
        """
        return self.decoder(z)
    
    def forward(self, x):
        """
        전체 VAE 순전파
        """
        # 1. 인코딩: x → (μ, log σ²)
        mu, logvar = self.encode(x)
        
        # 2. Reparameterization: (μ, log σ²) → z
        z = self.reparameterize(mu, logvar)
        
        # 3. 디코딩: z → x_reconstructed
        x_recon = self.decode(z)
        
        return x_recon, mu, logvar

# 2. ELBO 손실 함수 구현
def vae_loss(x_recon, x, mu, logvar, beta=1.0):
    """
    VAE의 ELBO 손실 함수
    
    ELBO = E[log p(x|z)] - β × KL[q(z|x) || p(z)]
    
    Args:
        x_recon: 재구성된 이미지
        x: 원본 이미지
        mu: 인코더가 출력한 평균
        logvar: 인코더가 출력한 log 분산
        beta: KL divergence 가중치 (β-VAE)
    """
    
    # 1. Reconstruction Loss: -E[log p(x|z)]
    # 베르누이 분포 가정 하에서 binary cross entropy 사용
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
    
    # 2. KL Divergence: KL[q(z|x) || p(z)]
    # q(z|x) = N(μ, σ²), p(z) = N(0, I)일 때의 해석적 해
    # KL = 0.5 * Σ(μ² + σ² - log σ² - 1)
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # 3. 총 손실 (ELBO의 음수)
    total_loss = recon_loss + beta * kl_divergence
    
    return total_loss, recon_loss, kl_divergence

# 3. 데이터 로더 설정
def get_data_loaders(batch_size=128):
    """
    MNIST 데이터셋 로더
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  # 28x28 → 784 평면화
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# 4. 훈련 함수
def train_vae(model, train_loader, optimizer, epoch, device, beta=1.0):
    """
    VAE 훈련 함수
    """
    model.train()
    train_loss = 0
    train_recon_loss = 0
    train_kl_loss = 0
    
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        
        # 순전파
        recon_data, mu, logvar = model(data)
        
        # 손실 계산
        loss, recon_loss, kl_loss = vae_loss(recon_data, data, mu, logvar, beta)
        
        # 역전파
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_recon_loss += recon_loss.item()
        train_kl_loss += kl_loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                  f'Loss: {loss.item() / len(data):.6f}')
    
    avg_loss = train_loss / len(train_loader.dataset)
    avg_recon_loss = train_recon_loss / len(train_loader.dataset)
    avg_kl_loss = train_kl_loss / len(train_loader.dataset)
    
    return avg_loss, avg_recon_loss, avg_kl_loss

# 5. 검증 함수
def test_vae(model, test_loader, device, beta=1.0):
    """
    VAE 검증 함수
    """
    model.eval()
    test_loss = 0
    test_recon_loss = 0
    test_kl_loss = 0
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            recon_data, mu, logvar = model(data)
            loss, recon_loss, kl_loss = vae_loss(recon_data, data, mu, logvar, beta)
            
            test_loss += loss.item()
            test_recon_loss += recon_loss.item()
            test_kl_loss += kl_loss.item()
    
    avg_loss = test_loss / len(test_loader.dataset)
    avg_recon_loss = test_recon_loss / len(test_loader.dataset)
    avg_kl_loss = test_kl_loss / len(test_loader.dataset)
    
    print(f'Test Loss: {avg_loss:.6f}, Recon Loss: {avg_recon_loss:.6f}, KL Loss: {avg_kl_loss:.6f}')
    
    return avg_loss, avg_recon_loss, avg_kl_loss

# 6. 시각화 함수들
def visualize_reconstruction(model, test_loader, device, num_samples=8):
    """
    원본과 재구성 이미지 비교
    """
    model.eval()
    with torch.no_grad():
        data, _ = next(iter(test_loader))
        data = data[:num_samples].to(device)
        recon_data, _, _ = model(data)
        
        # CPU로 이동 및 reshape
        data = data.cpu().view(num_samples, 28, 28)
        recon_data = recon_data.cpu().view(num_samples, 28, 28)
        
        fig, axes = plt.subplots(2, num_samples, figsize=(15, 4))
        
        for i in range(num_samples):
            # 원본 이미지
            axes[0, i].imshow(data[i], cmap='gray')
            axes[0, i].set_title('Original')
            axes[0, i].axis('off')
            
            # 재구성 이미지
            axes[1, i].imshow(recon_data[i], cmap='gray')
            axes[1, i].set_title('Reconstructed')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.show()

def generate_samples(model, device, num_samples=16):
    """
    잠재 공간에서 새로운 샘플 생성
    """
    model.eval()
    with torch.no_grad():
        # 표준 정규분포에서 잠재 벡터 샘플링
        z = torch.randn(num_samples, model.latent_size).to(device)
        
        # 디코더로 이미지 생성
        generated = model.decode(z).cpu().view(num_samples, 28, 28)
        
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        
        for i in range(num_samples):
            row, col = i // 4, i % 4
            axes[row, col].imshow(generated[i], cmap='gray')
            axes[row, col].axis('off')
        
        plt.suptitle('Generated Samples from Latent Space')
        plt.tight_layout()
        plt.show()

def visualize_latent_space(model, test_loader, device):
    """
    잠재 공간의 2D 시각화 (t-SNE 사용)
    """
    model.eval()
    latent_vectors = []
    labels = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            mu, _ = model.encode(data)
            latent_vectors.append(mu.cpu().numpy())
            labels.append(target.numpy())
            
            if len(latent_vectors) > 10:  # 계산 시간 단축을 위해 일부만 사용
                break
    
    latent_vectors = np.concatenate(latent_vectors, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    # t-SNE로 2D 축소
    tsne = TSNE(n_components=2, random_state=42)
    latent_2d = tsne.fit_transform(latent_vectors)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=labels, cmap='tab10')
    plt.colorbar(scatter)
    plt.title('Latent Space Visualization (t-SNE)')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.show()

def interpolate_latent_space(model, test_loader, device, steps=10):
    """
    잠재 공간에서의 보간 (interpolation)
    """
    model.eval()
    with torch.no_grad():
        # 두 개의 서로 다른 이미지 선택
        data, _ = next(iter(test_loader))
        img1, img2 = data[0:1].to(device), data[1:2].to(device)
        
        # 잠재 벡터 추출
        z1, _ = model.encode(img1)
        z2, _ = model.encode(img2)
        
        # 보간
        interpolated_images = []
        for i, alpha in enumerate(np.linspace(0, 1, steps)):
            z_interp = alpha * z2 + (1 - alpha) * z1
            img_interp = model.decode(z_interp).cpu().view(28, 28)
            interpolated_images.append(img_interp)
        
        fig, axes = plt.subplots(1, steps, figsize=(20, 2))
        for i, img in enumerate(interpolated_images):
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f'α={i/(steps-1):.1f}')
            axes[i].axis('off')
        
        plt.suptitle('Latent Space Interpolation')
        plt.tight_layout()
        plt.show()

# 7. β-VAE 실험
def beta_vae_experiment(model_class, train_loader, test_loader, device, betas=[0.1, 1.0, 4.0, 10.0]):
    """
    β-VAE 실험: 다양한 β 값에 따른 disentanglement 효과 비교
    """
    results = {}
    
    for beta in betas:
        print(f"\n=== Training β-VAE with β={beta} ===")
        
        # 모델 초기화
        model = model_class().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        # 간단한 훈련 (5 에포크)
        for epoch in range(1, 6):
            train_loss, recon_loss, kl_loss = train_vae(
                model, train_loader, optimizer, epoch, device, beta
            )
        
        # 결과 저장
        test_loss, test_recon, test_kl = test_vae(model, test_loader, device, beta)
        results[beta] = {
            'model': model,
            'test_loss': test_loss,
            'test_recon': test_recon,
            'test_kl': test_kl
        }
        
        # 생성 샘플 시각화
        print(f"Generated samples for β={beta}:")
        generate_samples(model, device, 16)
    
    return results

# 8. 메인 실행 함수
def main():
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 하이퍼파라미터 설정
    batch_size = 128
    learning_rate = 1e-3
    epochs = 20
    latent_size = 20
    beta = 1.0  # β-VAE 파라미터
    
    # 데이터 로더
    train_loader, test_loader = get_data_loaders(batch_size)
    
    # 모델 초기화
    model = VAE(latent_size=latent_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 훈련 기록용
    train_losses = []
    train_recon_losses = []
    train_kl_losses = []
    test_losses = []
    
    print("Starting VAE training...")
    
    # 훈련 루프
    for epoch in range(1, epochs + 1):
        # 훈련
        train_loss, train_recon, train_kl = train_vae(
            model, train_loader, optimizer, epoch, device, beta
        )
        
        # 검증
        test_loss, test_recon, test_kl = test_vae(model, test_loader, device, beta)
        
        # 기록
        train_losses.append(train_loss)
        train_recon_losses.append(train_recon)
        train_kl_losses.append(train_kl)
        test_losses.append(test_loss)
        
        # 주기적으로 결과 시각화
        if epoch % 5 == 0:
            print(f"\nEpoch {epoch} Results:")
            visualize_reconstruction(model, test_loader, device)
            generate_samples(model, device)
    
    # 훈련 과정 시각화
    plot_training_curves(train_losses, train_recon_losses, train_kl_losses, test_losses)
    
    # 최종 결과 시각화
    print("\nFinal Results:")
    visualize_reconstruction(model, test_loader, device)
    generate_samples(model, device)
    visualize_latent_space(model, test_loader, device)
    interpolate_latent_space(model, test_loader, device)
    
    # β-VAE 실험
    print("\n=== β-VAE Experiment ===")
    beta_results = beta_vae_experiment(VAE, train_loader, test_loader, device)
    
    return model, beta_results

def plot_training_curves(train_losses, train_recon_losses, train_kl_losses, test_losses):
    """
    훈련 과정 시각화
    """
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(15, 5))
    
    # 전체 손실
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, test_losses, 'r-', label='Test Loss')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 재구성 손실
    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_recon_losses, 'g-', label='Reconstruction Loss')
    plt.title('Reconstruction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # KL 발산
    plt.subplot(1, 3, 3)
    plt.plot(epochs, train_kl_losses, 'm-', label='KL Divergence')
    plt.title('KL Divergence')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# 9. 고급 VAE 변형들
class ConditionalVAE(nn.Module):
    """
    조건부 VAE (CVAE): 클래스 라벨을 조건으로 하는 VAE
    """
    def __init__(self, input_size=784, hidden_size=400, latent_size=20, num_classes=10):
        super(ConditionalVAE, self).__init__()
        
        self.num_classes = num_classes
        self.latent_size = latent_size
        
        # 인코더 (x + y → z)
        self.encoder = nn.Sequential(
            nn.Linear(input_size + num_classes, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(hidden_size, latent_size)
        self.fc_logvar = nn.Linear(hidden_size, latent_size)
        
        # 디코더 (z + y → x)
        self.decoder = nn.Sequential(
            nn.Linear(latent_size + num_classes, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()
        )
    
    def encode(self, x, y):
        # 원-핫 인코딩
        y_onehot = F.one_hot(y, self.num_classes).float()
        # x와 y 연결
        xy = torch.cat([x, y_onehot], dim=1)
        
        h = self.encoder(xy)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def decode(self, z, y):
        # 원-핫 인코딩
        y_onehot = F.one_hot(y, self.num_classes).float()
        # z와 y 연결
        zy = torch.cat([z, y_onehot], dim=1)
        return self.decoder(zy)
    
    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, y)
        return x_recon, mu, logvar

class BetaVAE(VAE):
    """
    β-VAE: disentangled representation learning을 위한 VAE
    """
    def __init__(self, input_size=784, hidden_size=400, latent_size=20, beta=4.0):
        super(BetaVAE, self).__init__(input_size, hidden_size, latent_size)
        self.beta = beta
    
    def loss_function(self, x_recon, x, mu, logvar):
        # β를 내부적으로 적용
        return vae_loss(x_recon, x, mu, logvar, self.beta)

# 10. 잠재 공간 분석 도구들
def compute_disentanglement_metric(model, test_loader, device, num_samples=1000):
    """
    Disentanglement 정도를 측정하는 간단한 메트릭
    """
    model.eval()
    latent_codes = []
    labels = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            mu, _ = model.encode(data)
            latent_codes.append(mu.cpu())
            labels.append(target)
            
            if len(latent_codes) * data.size(0) >= num_samples:
                break
    
    latent_codes = torch.cat(latent_codes, dim=0)[:num_samples]
    labels = torch.cat(labels, dim=0)[:num_samples]
    
    # 각 잠재 차원별로 클래스 간 분산과 클래스 내 분산 계산
    disentanglement_scores = []
    
    for dim in range(latent_codes.size(1)):
        latent_dim = latent_codes[:, dim]
        
        # 클래스별 평균 계산
        class_means = []
        for class_id in range(10):  # MNIST는 10개 클래스
            class_mask = (labels == class_id)
            if class_mask.sum() > 0:
                class_mean = latent_dim[class_mask].mean()
                class_means.append(class_mean)
        
        if len(class_means) > 1:
            # 클래스 간 분산 vs 전체 분산의 비율
            class_means = torch.tensor(class_means)
            between_class_var = class_means.var()
            total_var = latent_dim.var()
            
            if total_var > 0:
                disentanglement_score = between_class_var / total_var
                disentanglement_scores.append(disentanglement_score.item())
    
    return np.mean(disentanglement_scores) if disentanglement_scores else 0.0

def analyze_latent_traversal(model, device, latent_dim=0, num_steps=10, range_scale=3.0):
    """
    특정 잠재 차원을 따라 이동하면서 생성되는 이미지 변화 분석
    """
    model.eval()
    
    # 기준 잠재 벡터 (평균)
    base_z = torch.zeros(1, model.latent_size).to(device)
    
    # 지정된 차원을 따라 이동
    traversal_range = torch.linspace(-range_scale, range_scale, num_steps)
    generated_images = []
    
    with torch.no_grad():
        for value in traversal_range:
            z = base_z.clone()
            z[0, latent_dim] = value
            
            generated = model.decode(z).cpu().view(28, 28)
            generated_images.append(generated)
    
    # 시각화
    fig, axes = plt.subplots(1, num_steps, figsize=(20, 2))
    for i, img in enumerate(generated_images):
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'{traversal_range[i]:.1f}')
        axes[i].axis('off')
    
    plt.suptitle(f'Latent Dimension {latent_dim} Traversal')
    plt.tight_layout()
    plt.show()

# 11. 실제 실행 코드
if __name__ == "__main__":
    # 시드 설정 (재현가능한 결과를 위해)
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 메인 실행
    trained_model, beta_results = main()
    
    # 추가 분석
    print("\n=== Advanced Analysis ===")
    
    # Disentanglement 분석
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, test_loader = get_data_loaders()
    
    disentanglement_score = compute_disentanglement_metric(trained_model, test_loader, device)
    print(f"Disentanglement Score: {disentanglement_score:.4f}")
    
    # 잠재 공간 순회 분석
    print("\nLatent space traversal for different dimensions:")
    for dim in range(min(5, trained_model.latent_size)):
        analyze_latent_traversal(trained_model, device, latent_dim=dim)
    
    print("\nVAE training and analysis completed!")

"""
실습 과제:

1. 기본 VAE 구현 및 훈련
   - MNIST 데이터셋으로 VAE 훈련
   - 재구성 품질과 생성 품질 평가

2. β-VAE 실험
   - β = [0.1, 1.0, 4.0, 10.0]으로 실험
   - β 값에 따른 재구성 품질 vs disentanglement 트레이드오프 관찰

3. 잠재 공간 분석
   - t-SNE로 잠재 공간 시각화
   - 잠재 차원별 순회 분석
   - Disentanglement 정도 정량 평가

4. 조건부 VAE (도전 과제)
   - 클래스 라벨을 조건으로 하는 CVAE 구현
   - 특정 숫자를 생성하도록 제어

5. 손실 함수 분석
   - 훈련 과정에서 재구성 손실과 KL 발산의 변화 관찰
   - ELBO의 각 항이 모델 성능에 미치는 영향 분석
"""