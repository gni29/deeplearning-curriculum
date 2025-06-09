"""
U-Net 실습: 의료 영상 분할을 위한 완전한 파이프라인
데이터 전처리부터 모델 구현, 훈련, 평가까지

이 실습에서는 다음을 다룹니다:
1. 의료 영상 데이터 전처리 (정규화, 증강, 검증)
2. U-Net 아키텍처 완전 구현
3. 의료 영상 특화 손실 함수
4. 평가 지표 및 시각화
5. 실제 의료 데이터셋 활용
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from PIL import Image
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 1. 의료 영상 데이터 전처리 클래스
class MedicalImagePreprocessor:
    """
    의료 영상 전처리를 위한 통합 클래스
    다양한 정규화 기법과 통계적 검증 방법 제공
    """
    
    def __init__(self, target_size=(256, 256)):
        self.target_size = target_size
        self.stats = {}
    
    def analyze_dataset_statistics(self, image_paths):
        """
        데이터셋의 통계적 특성 분석
        - 픽셀 강도 분포
        - 이미지 크기 분포
        - 대비 및 밝기 통계
        """
        print("Analyzing dataset statistics...")
        
        pixel_values = []
        image_sizes = []
        contrasts = []
        brightnesses = []
        
        for img_path in tqdm(image_paths[:100]):  # 샘플링으로 속도 향상
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # 픽셀 값 수집
                pixel_values.extend(img.flatten())
                
                # 이미지 크기
                image_sizes.append(img.shape)
                
                # 대비 (표준편차)
                contrasts.append(np.std(img))
                
                # 밝기 (평균)
                brightnesses.append(np.mean(img))
        
        # 통계 저장
        pixel_values = np.array(pixel_values)
        self.stats = {
            'pixel_mean': np.mean(pixel_values),
            'pixel_std': np.std(pixel_values),
            'pixel_min': np.min(pixel_values),
            'pixel_max': np.max(pixel_values),
            'pixel_median': np.median(pixel_values),
            'common_size': max(set(image_sizes), key=image_sizes.count),
            'contrast_mean': np.mean(contrasts),
            'contrast_std': np.std(contrasts),
            'brightness_mean': np.mean(brightnesses),
            'brightness_std': np.std(brightnesses)
        }
        
        self.visualize_statistics(pixel_values, contrasts, brightnesses)
        return self.stats
    
    def visualize_statistics(self, pixel_values, contrasts, brightnesses):
        """
        데이터셋 통계 시각화
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 픽셀 강도 히스토그램
        axes[0, 0].hist(pixel_values[::1000], bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Pixel Intensity Distribution')
        axes[0, 0].set_xlabel('Pixel Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(self.stats['pixel_mean'], color='red', linestyle='--', label=f'Mean: {self.stats["pixel_mean"]:.1f}')
        axes[0, 0].legend()
        
        # 대비 분포
        axes[0, 1].hist(contrasts, bins=30, alpha=0.7, color='green', edgecolor='black')
        axes[0, 1].set_title('Image Contrast Distribution')
        axes[0, 1].set_xlabel('Standard Deviation')
        axes[0, 1].set_ylabel('Frequency')
        
        # 밝기 분포
        axes[1, 0].hist(brightnesses, bins=30, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 0].set_title('Image Brightness Distribution')
        axes[1, 0].set_xlabel('Mean Pixel Value')
        axes[1, 0].set_ylabel('Frequency')
        
        # 통계 요약 텍스트
        stats_text = f"""Dataset Statistics:
        Pixel Mean: {self.stats['pixel_mean']:.2f}
        Pixel Std: {self.stats['pixel_std']:.2f}
        Pixel Range: [{self.stats['pixel_min']}, {self.stats['pixel_max']}]
        Common Size: {self.stats['common_size']}
        Avg Contrast: {self.stats['contrast_mean']:.2f}
        Avg Brightness: {self.stats['brightness_mean']:.2f}
        """
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def normalize_image(self, image, method='z_score'):
        """
        다양한 정규화 방법 적용
        
        Args:
            image: 입력 이미지
            method: 정규화 방법
                - 'z_score': Z-score 정규화 (평균 0, 표준편차 1)
                - 'min_max': Min-Max 정규화 (0-1 범위)
                - 'robust': Robust 정규화 (중앙값과 IQR 사용)
                - 'histogram_eq': 히스토그램 평활화
                - 'clahe': CLAHE (대비 제한 적응 히스토그램 평활화)
        """
        if method == 'z_score':
            # Z-score 정규화: (x - μ) / σ
            mean = np.mean(image)
            std = np.std(image)
            if std > 0:
                normalized = (image - mean) / std
            else:
                normalized = image - mean
            # [0, 1] 범위로 스케일링
            normalized = (normalized - normalized.min()) / (normalized.max() - normalized.min() + 1e-8)
            
        elif method == 'min_max':
            # Min-Max 정규화: (x - min) / (max - min)
            min_val = np.min(image)
            max_val = np.max(image)
            if max_val > min_val:
                normalized = (image - min_val) / (max_val - min_val)
            else:
                normalized = np.zeros_like(image)
                
        elif method == 'robust':
            # Robust 정규화: (x - median) / IQR
            median = np.median(image)
            q75, q25 = np.percentile(image, [75, 25])
            iqr = q75 - q25
            if iqr > 0:
                normalized = (image - median) / iqr
            else:
                normalized = image - median
            # [0, 1] 범위로 스케일링
            normalized = (normalized - normalized.min()) / (normalized.max() - normalized.min() + 1e-8)
            
        elif method == 'histogram_eq':
            # 히스토그램 평활화
            image_uint8 = (image * 255).astype(np.uint8)
            normalized = cv2.equalizeHist(image_uint8) / 255.0
            
        elif method == 'clahe':
            # CLAHE (Contrast Limited Adaptive Histogram Equalization)
            image_uint8 = (image * 255).astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            normalized = clahe.apply(image_uint8) / 255.0
            
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return normalized.astype(np.float32)
    
    def validate_preprocessing(self, original, preprocessed, method_name):
        """
        전처리 결과의 통계적 검증
        """
        print(f"\n=== Validation for {method_name} ===")
        
        # 기본 통계
        print(f"Original - Mean: {np.mean(original):.3f}, Std: {np.std(original):.3f}")
        print(f"Processed - Mean: {np.mean(preprocessed):.3f}, Std: {np.std(preprocessed):.3f}")
        
        # 정보 보존 검증 (상관계수)
        correlation = np.corrcoef(original.flatten(), preprocessed.flatten())[0, 1]
        print(f"Correlation with original: {correlation:.3f}")
        
        # 동적 범위 검증
        original_range = np.max(original) - np.min(original)
        processed_range = np.max(preprocessed) - np.min(preprocessed)
        print(f"Dynamic range - Original: {original_range:.3f}, Processed: {processed_range:.3f}")
        
        # 엔트로피 변화 (정보량 측정)
        def calculate_entropy(img):
            hist, _ = np.histogram(img, bins=256, range=(0, 1))
            hist = hist / hist.sum()
            hist = hist[hist > 0]  # 0인 빈을 제거
            return -np.sum(hist * np.log2(hist))
        
        original_entropy = calculate_entropy(original)
        processed_entropy = calculate_entropy(preprocessed)
        print(f"Entropy - Original: {original_entropy:.3f}, Processed: {processed_entropy:.3f}")
        
        return {
            'correlation': correlation,
            'entropy_change': processed_entropy - original_entropy,
            'range_preservation': processed_range / original_range if original_range > 0 else 0
        }

# 2. 의료 영상 데이터셋 클래스
class MedicalSegmentationDataset(Dataset):
    """
    의료 영상 분할을 위한 데이터셋 클래스
    """
    
    def __init__(self, image_paths, mask_paths, preprocessor, augmentations=None, is_training=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.preprocessor = preprocessor
        self.augmentations = augmentations
        self.is_training = is_training
        
        # 기본 변환 (크기 조정)
        self.base_transform = A.Compose([
            A.Resize(preprocessor.target_size[0], preprocessor.target_size[1]),
            A.Normalize(mean=0.0, std=1.0),  # 이미 전처리된 이미지 가정
            ToTensorV2()
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 이미지 로드
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        # 정규화
        image = image / 255.0
        image = self.preprocessor.normalize_image(image, method='clahe')
        
        # 마스크 이진화
        mask = (mask > 127).astype(np.float32)
        
        # 데이터 증강 적용
        if self.augmentations is not None and self.is_training:
            augmented = self.augmentations(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # 기본 변환 적용
        transformed = self.base_transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']
        
        # 채널 차원 추가 (그레이스케일 → 1채널)
        if len(image.shape) == 2:
            image = image.unsqueeze(0)
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)
        
        return image, mask

# 3. 의료 영상용 데이터 증강
def get_medical_augmentations():
    """
    의료 영상에 적합한 데이터 증강 기법
    - 기하학적 변환은 신중하게 적용
    - 의료 영상의 특성을 보존하는 증강 우선
    """
    return A.Compose([
        # 기하학적 변환 (의료 영상에서 신중하게)
        A.HorizontalFlip(p=0.5),  # 좌우 대칭성이 있는 경우만
        A.Rotate(limit=10, p=0.3),  # 작은 각도로 제한
        
        # 강도 변환 (의료 영상에 중요)
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        A.GaussianNoise(var_limit=(0.001, 0.01), p=0.3),
        
        # 블러링 (실제 촬영 조건 모사)
        A.OneOf([
            A.GaussianBlur(blur_limit=3, p=0.5),
            A.MotionBlur(blur_limit=3, p=0.5),
        ], p=0.2),
        
        # 탄성 변형 (조직의 자연스러운 변형)
        A.ElasticTransform(alpha=50, sigma=5, alpha_affine=5, p=0.2),
        
        # 그리드 왜곡 (미세한 기하학적 변형)
        A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.2),
    ])

# 4. U-Net 아키텍처 구현
class DoubleConv(nn.Module):
    """
    U-Net의 기본 블록: Conv2d → BatchNorm → ReLU → Conv2d → BatchNorm → ReLU
    """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """
    다운샘플링: MaxPool2d → DoubleConv
    """
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """
    업샘플링: ConvTranspose2d → Concat → DoubleConv
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        # x1: 업샘플링할 특징 맵 (디코더)
        # x2: 연결할 특징 맵 (인코더, skip connection)
        
        x1 = self.up(x1)
        
        # 크기가 다를 경우 패딩으로 맞춤
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # 채널 차원에서 연결
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """
    최종 출력 레이어
    """
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """
    완전한 U-Net 구현
    """
    def __init__(self, n_channels=1, n_classes=1, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # 인코더 경로
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        # 디코더 경로
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        # 최종 출력
        self.outc = OutConv(64, n_classes)
    
    def forward(self, x):
        # 인코더
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # 디코더 (skip connections 포함)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # 최종 출력
        logits = self.outc(x)
        return logits

# 5. 의료 영상 특화 손실 함수
class DiceLoss(nn.Module):
    """
    Dice Loss: 분할 작업에서 클래스 불균형 문제 해결
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        # Sigmoid 적용 (확률로 변환)
        inputs = torch.sigmoid(inputs)
        
        # 평면화
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Dice 계수 계산
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice

class FocalLoss(nn.Module):
    """
    Focal Loss: 어려운 예제에 더 집중하는 손실 함수
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class CombinedLoss(nn.Module):
    """
    여러 손실 함수의 가중 결합
    """
    def __init__(self, dice_weight=0.5, focal_weight=0.3, bce_weight=0.2):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.bce_weight = bce_weight
    
    def forward(self, inputs, targets):
        dice = self.dice_loss(inputs, targets)
        focal = self.focal_loss(inputs, targets)
        bce = self.bce_loss(inputs, targets)
        
        total_loss = (self.dice_weight * dice + 
                     self.focal_weight * focal + 
                     self.bce_weight * bce)
        
        return total_loss, dice, focal, bce

# 6. 평가 지표
class SegmentationMetrics:
    """
    분할 작업을 위한 종합적인 평가 지표
    """
    
    @staticmethod
    def calculate_iou(pred, target, threshold=0.5, smooth=1e-6):
        """
        Intersection over Union (IoU) 계산
        """
        pred = (pred > threshold).float()
        target = target.float()
        
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        
        iou = (intersection + smooth) / (union + smooth)
        return iou.item()
    
    @staticmethod
    def calculate_dice(pred, target, threshold=0.5, smooth=1e-6):
        """
        Dice 계수 계산
        """
        pred = (pred > threshold).float()
        target = target.float()
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        
        return dice.item()
    
    @staticmethod
    def calculate_pixel_accuracy(pred, target, threshold=0.5):
        """
        픽셀 정확도 계산
        """
        pred = (pred > threshold).float()
        target = target.float()
        
        correct = (pred == target).float().sum()
        total = target.numel()
        
        return (correct / total).item()
    
    @staticmethod
    def calculate_sensitivity_specificity(pred, target, threshold=0.5):
        """
        민감도(Sensitivity)와 특이도(Specificity) 계산
        """
        pred = (pred > threshold).float().view(-1)
        target = target.float().view(-1)
        
        # True Positive, False Positive, True Negative, False Negative
        tp = ((pred == 1) & (target == 1)).sum().float()
        fp = ((pred == 1) & (target == 0)).sum().float()
        tn = ((pred == 0) & (target == 0)).sum().float()
        fn = ((pred == 0) & (target == 1)).sum().float()
        
        sensitivity = tp / (tp + fn + 1e-6)  # Recall
        specificity = tn / (tn + fp + 1e-6)
        
        return sensitivity.item(), specificity.item()
    
    @staticmethod
    def calculate_hausdorff_distance(pred, target, threshold=0.5):
        """
        Hausdorff Distance 계산 (경계 정확도 측정)
        """
        try:
            from scipy.spatial.distance import directed_hausdorff
            
            pred = (pred > threshold).cpu().numpy()
            target = target.cpu().numpy()
            
            # 경계 픽셀 추출
            pred_contour = cv2.Canny((pred * 255).astype(np.uint8), 50, 150)
            target_contour = cv2.Canny((target * 255).astype(np.uint8), 50, 150)
            
            # 좌표 추출
            pred_points = np.column_stack(np.where(pred_contour > 0))
            target_points = np.column_stack(np.where(target_contour > 0))
            
            if len(pred_points) == 0 or len(target_points) == 0:
                return float('inf')
            
            # Hausdorff Distance 계산
            hd1 = directed_hausdorff(pred_points, target_points)[0]
            hd2 = directed_hausdorff(target_points, pred_points)[0]
            
            return max(hd1, hd2)
        
        except ImportError:
            print("scipy not available for Hausdorff distance calculation")
            return None

# 7. 훈련 함수
def train_unet(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
    """
    U-Net 훈련 함수
    """
    train_losses = []
    val_losses = []
    val_ious = []
    val_dices = []
    
    best_iou = 0.0
    
    for epoch in range(num_epochs):
        # 훈련 모드
        model.train()
        train_loss = 0.0
        train_dice_loss = 0.0
        train_focal_loss = 0.0
        train_bce_loss = 0.0
        
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 50)
        
        # 훈련 루프
        for batch_idx, (images, masks) in enumerate(tqdm(train_loader, desc='Training')):
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            
            # 순전파
            outputs = model(images)
            
            # 손실 계산
            if isinstance(criterion, CombinedLoss):
                loss, dice_loss, focal_loss, bce_loss = criterion(outputs, masks)
                train_dice_loss += dice_loss.item()
                train_focal_loss += focal_loss.item()
                train_bce_loss += bce_loss.item()
            else:
                loss = criterion(outputs, masks)
            
            # 역전파
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 검증
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_dice = 0.0
        
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc='Validation'):
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                
                if isinstance(criterion, CombinedLoss):
                    loss, _, _, _ = criterion(outputs, masks)
                else:
                    loss = criterion(outputs, masks)
                
                val_loss += loss.item()
                
                # 평가 지표 계산
                probs = torch.sigmoid(outputs)
                val_iou += SegmentationMetrics.calculate_iou(probs, masks)
                val_dice += SegmentationMetrics.calculate_dice(probs, masks)
        
        # 평균 계산
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_iou /= len(val_loader)
        val_dice /= len(val_loader)
        
        # 기록
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_ious.append(val_iou)
        val_dices.append(val_dice)
        
        # 출력
        print(f'Train Loss: {train_loss:.4f}')
        if isinstance(criterion, CombinedLoss):
            print(f'  - Dice: {train_dice_loss/len(train_loader):.4f}')
            print(f'  - Focal: {train_focal_loss/len(train_loader):.4f}')
            print(f'  - BCE: {train_bce_loss/len(train_loader):.4f}')
        print(f'Val Loss: {val_loss:.4f}')
        print(f'Val IoU: {val_iou:.4f}')
        print(f'Val Dice: {val_dice:.4f}')
        
        # 스케줄러 업데이트
        if scheduler:
            scheduler.step(val_loss)
        
        # 최고 모델 저장
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), 'best_unet_model.pth')
            print(f'New best model saved with IoU: {best_iou:.4f}')
    
    return train_losses, val_losses, val_ious, val_dices

# 8. 시각화 함수들
def visualize_predictions(model, data_loader, device, num_samples=4):
    """
    예측 결과 시각화
    """
    model.eval()
    
    with torch.no_grad():
        for images, masks in data_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            # CPU로 이동
            images = images.cpu()
            masks = masks.cpu()
            preds = preds.cpu()
            probs = probs.cpu()
            
            fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
            
            for i in range(min(num_samples, images.size(0))):
                # 원본 이미지
                axes[i, 0].imshow(images[i, 0], cmap='gray')
                axes[i, 0].set_title('Original Image')
                axes[i, 0].axis('off')
                
                # 실제 마스크
                axes[i, 1].imshow(masks[i, 0], cmap='gray')
                axes[i, 1].set_title('Ground Truth')
                axes[i, 1].axis('off')
                
                # 예측 확률
                axes[i, 2].imshow(probs[i, 0], cmap='hot', vmin=0, vmax=1)
                axes[i, 2].set_title('Prediction Probability')
                axes[i, 2].axis('off')
                
                # 이진 예측
                axes[i, 3].imshow(preds[i, 0], cmap='gray')
                axes[i, 3].set_title('Binary Prediction')
                axes[i, 3].axis('off')
                
                # IoU 계산 및 표시
                iou = SegmentationMetrics.calculate_iou(probs[i:i+1], masks[i:i+1])
                dice = SegmentationMetrics.calculate_dice(probs[i:i+1], masks[i:i+1])
                
                fig.suptitle(f'Sample {i+1} - IoU: {iou:.3f}, Dice: {dice:.3f}', 
                           fontsize=14, y=0.98)
            
            plt.tight_layout()
            plt.show()
            break

def plot_training_history(train_losses, val_losses, val_ious, val_dices):
    """
    훈련 과정 시각화
    """
    epochs = range(1, len(train_losses) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 손실
    axes[0, 0].plot(epochs, train_losses, 'b-', label='Train Loss')
    axes[0, 0].plot(epochs, val_losses, 'r-', label='Val Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # IoU
    axes[0, 1].plot(epochs, val_ious, 'g-', label='Val IoU')
    axes[0, 1].set_title('Validation IoU')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('IoU')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Dice
    axes[1, 0].plot(epochs, val_dices, 'm-', label='Val Dice')
    axes[1, 0].set_title('Validation Dice Score')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Dice')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 모든 지표 비교
    # 정규화된 지표들
    norm_train_loss = [(1 - loss/max(train_losses)) for loss in train_losses]
    norm_val_loss = [(1 - loss/max(val_losses)) for loss in val_losses]
    
    axes[1, 1].plot(epochs, norm_train_loss, 'b--', label='Norm Train Loss (inverted)')
    axes[1, 1].plot(epochs, norm_val_loss, 'r--', label='Norm Val Loss (inverted)')
    axes[1, 1].plot(epochs, val_ious, 'g-', label='Val IoU')
    axes[1, 1].plot(epochs, val_dices, 'm-', label='Val Dice')
    axes[1, 1].set_title('All Metrics Comparison')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Normalized Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()

# 9. 메인 실행 함수
def main():
    """
    전체 파이프라인 실행
    """
    # 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 하이퍼파라미터
    batch_size = 8
    learning_rate = 1e-4
    num_epochs = 50
    image_size = (256, 256)
    
    # 데이터 경로 (실제 사용 시 수정 필요)
    # train_image_paths = glob.glob('data/train/images/*.png')
    # train_mask_paths = glob.glob('data/train/masks/*.png')
    # val_image_paths = glob.glob('data/val/images/*.png')
    # val_mask_paths = glob.glob('data/val/masks/*.png')
    
    # 더미 데이터로 예시 (실제 데이터로 교체)
    print("Creating dummy dataset for demonstration...")
    train_image_paths = ['dummy_image.png'] * 100
    train_mask_paths = ['dummy_mask.png'] * 100
    val_image_paths = ['dummy_image.png'] * 20
    val_mask_paths = ['dummy_mask.png'] * 20
    
    # 전처리기 초기화
    preprocessor = MedicalImagePreprocessor(target_size=image_size)
    
    # 데이터셋 분석 (실제 데이터가 있을 때)
    # stats = preprocessor.analyze_dataset_statistics(train_image_paths)
    
    # 데이터 증강
    train_augmentations = get_medical_augmentations()
    
    # 데이터셋 생성
    train_dataset = MedicalSegmentationDataset(
        train_image_paths, train_mask_paths, preprocessor, train_augmentations, is_training=True
    )
    val_dataset = MedicalSegmentationDataset(
        val_image_paths, val_mask_paths, preprocessor, None, is_training=False
    )
    
    # 데이터 로더
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # 모델 초기화
    model = UNet(n_channels=1, n_classes=1, bilinear=True).to(device)
    
    # 손실 함수 및 옵티마이저
    criterion = CombinedLoss(dice_weight=0.4, focal_weight=0.4, bce_weight=0.2)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # 훈련
    print("\nStarting training...")
    train_losses, val_losses, val_ious, val_dices = train_unet(
        model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device
    )
    
    # 결과 시각화
    plot_training_history(train_losses, val_losses, val_ious, val_dices)
    
    # 예측 시각화
    print("\nVisualizing predictions...")
    visualize_predictions(model, val_loader, device)
    
    return model

if __name__ == "__main__":
    model = main()
    print("U-Net training completed!")

"""
실습 과제:

1. 데이터 전처리 비교 실험
   - 다양한 정규화 방법 (z_score, min_max, clahe) 성능 비교
   - 전처리 전후 통계 분석 및 검증

2. 데이터 증강 효과 분석
   - 증강 없음 vs 기본 증강 vs 의료 특화 증강 비교
   - 각 증강 기법의 개별 효과 측정

3. 손실 함수 실험
   - BCE vs Dice vs Focal vs Combined Loss 비교
   - 각 손실 함수의 가중치 조정 실험

4. 모델 아키텍처 비교
   - 기본 U-Net vs Attention U-Net vs U-Net++ 성능 비교
   - 다양한 깊이와 채널 수 실험

5. 평가 지표 분석
   - IoU, Dice, Hausdorff Distance 등 다양한 지표로 성능 평가
   - 의료 영상 특성에 따른 적절한 지표 선택

6. 실제 데이터셋 적용
   - ISIC (피부암), BraTS (뇌종양), 또는 다른 의료 데이터셋 사용
   - 도메인 특화 전처리 및 후처리 기법 적용
"""