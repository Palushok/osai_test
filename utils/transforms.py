import albumentations as alb
from albumentations.pytorch.transforms import ToTensorV2

def valid_transform(h, w):
    return alb.Compose([
        alb.Resize(h, w),
        alb.Normalize(),
        ToTensorV2()
    ])

def train_transform(h, w):
    return alb.Compose([
        alb.OneOf([
            alb.Downscale(scale_min=0.4, scale_max=0.9, p=0.3),
            alb.ImageCompression(quality_lower=10, quality_upper=95, p=0.7),
        ]),
        alb.HorizontalFlip(p=0.5),
        alb.GaussNoise(p=0.3),
        alb.OneOf([
            alb.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=50,
                val_shift_limit=40, p=0.3),
            alb.RGBShift(p=0.3),
        ]),
        alb.RandomBrightness(0.1, p=0.3),
        alb.RandomFog(
            fog_coef_lower=0.05, fog_coef_upper=0.11,
            alpha_coef=0.3, p=0.2),
        alb.ToGray(p=0.1),
        alb.Rotate(4, p=0.1),
        alb.Resize(h, w),
        alb.Normalize(),
        ToTensorV2()
    ])