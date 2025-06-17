##
## EPITECH PROJECT, 2025
## MaskGenerator
## File description:
## augmentation_factory
##

import albumentations as A

class BlurTransform:
    def __init__(self):
        self.transform = A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=0.4),
            A.MedianBlur(blur_limit=5, p=0.2),
            A.MotionBlur(blur_limit=(3, 7), p=0.4),
        ], p=0.3)

    def __call__(self):
        return self.transform

class NoiseTransform:
    def __init__(self):
        self.transform = A.OneOf([
            A.GaussNoise(std_range=(0.1, 0.2), p=0.25),
            A.ISONoise(color_shift=(0.01, 0.5), intensity=(0.1, 0.5), p=0.3),
            A.MultiplicativeNoise(multiplier=(0.7, 1.5), elementwise=True, p=0.2),
            A.SaltAndPepper(amount=(0.02, 0.1), p=0.25),
        ], p=0.3)

    def __call__(self):
        return self.transform

class DropoutTransform:
    def __init__(self):
        self.transform = A.OneOf([
            A.OneOf([
                A.ConstrainedCoarseDropout(num_holes_range=(1, 4), hole_height_range=(0.2, 0.7), hole_width_range=(0.2, 0.7), fill="random", mask_indices=[1.0], p=0.33),
                A.ConstrainedCoarseDropout(num_holes_range=(1, 4), hole_height_range=(0.2, 0.7), hole_width_range=(0.2, 0.7), fill=0, mask_indices=[1.0], p=0.33),
                A.ConstrainedCoarseDropout(num_holes_range=(1, 4), hole_height_range=(0.2, 0.7), hole_width_range=(0.2, 0.7), fill="random_uniform", mask_indices=[1.0], p=0.33),
            ], p=0.5),
            A.OneOf([
                A.GridDropout(unit_size_range=(20, 50), fill=0, p=0.33),
                A.GridDropout(unit_size_range=(20, 50), fill="random_uniform", p=0.33),
            ], p=0.5)
        ], p=0.3)

    def __call__(self):
        return self.transform

class GeometryTransform:
    def __init__(self):
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Affine(
                rotate=(-10, 10),     # petites rotations
                translate_percent=(-0.05, 0.05),  # petits déplacements
                scale=(0.9, 1.1),     # zoom léger
                shear=(-5, 5),        # léger cisaillement
                p=0.4
            ),
            A.Perspective(scale=(0.02, 0.05), p=0.3),
        ], p=0.7)

    def __call__(self):
        return self.transform

class ColorInvarianceTransform:
    def __init__(self):
        self.transform = A.OneOf([
            A.ToGray(p=0.5),
            A.OneOf([
                A.ChannelDropout(channel_drop_range=(1, 2), fill=0, p=0.5),
                A.ChannelDropout(channel_drop_range=(1, 1), fill=200, p=0.5),
            ])
        ], p=0.2)

    def __call__(self):
        return self.transform

class ColorVariationTransform:
    def __init__(self):
        self.transform = A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.6, contrast_limit=0.3, p=1.0),
            A.ColorJitter(brightness=0.6, contrast=0.2, saturation=0.2, hue=0.3, p=1.0),
            A.HueSaturationValue(hue_shift_limit=100, sat_shift_limit=125, val_shift_limit=100, p=1.0),
        ], p=0.7)

    def __call__(self):
        return self.transform

class WeatherTransform:
    def __init__(self):
        self.transform = A.OneOf([
            A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), num_flare_circles_range=(6, 7), src_radius=400, method="physics_based", p=0.40),
            A.RandomShadow(shadow_roi=(0, 0.3, 1, 1), num_shadows_limit=(1, 2), shadow_intensity_range=(0.5, 0.8), p=0.20),
            A.RandomRain(slant_range=(-10, 10), drop_length=50, drop_width=1, rain_type="default", p=0.20),
            A.RandomSnow(brightness_coeff=2.5, snow_point_range=(0.1, 0.3), p=0.20)
        ], p=0.3)

    def __call__(self):
        return self.transform

class AugmentationFactory:
    def __init__(self):
        self.transformations = {
            "blur": BlurTransform(),
            "noise": NoiseTransform(),
            "dropout": DropoutTransform(),
            "geometry": GeometryTransform(),
            "color_invariance": ColorInvarianceTransform(),
            "color_variation": ColorVariationTransform(),
            "weather": WeatherTransform(),
        }

    def build(self, augmentation_names):
        transforms = []
        for name in augmentation_names:
            if name in self.transformations:
                transforms.append(self.transformations[name]())
            else:
                raise ValueError(f"Unknown augmentation: {name}")

        return A.Compose(transforms)
