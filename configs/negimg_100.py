import os
import albumentations as A
import albumentations.augmentations.functional as F
from albumentations.pytorch import ToTensorV2

class Config():
    # parser.add_argument("--organ_id", help="ID of the organ to train a model for", type=int)
    # parser.add_argument("--data_dir", help="Path to DSAD dataset")
    # parser.add_argument("--output_dir", help="Path to output folder")
    # parser.add_argument("--segformer", help="Specifity to use SegFormer instead of DeepLabV3", action="store_true")
    # parser.add_argument("--unet", help="Specifity to use UNet instead of DeepLabV3"

    ORGAN_ID = 4
    DATA_DIR = r"/scratch.global/laparoscopic_segmentation/Dresden/"
    OUTPUT_DIR = "."
    SEGFORMER = False
    UNET = False
    P_NEG_IMG = 1

    MIXED_PRECISION = True
    EPOCHS = 100
    LR = 1e-6
 
    AUG_ID = 1 # rand_pair
    K = 0.1
    AUGS = ["none", "rand_pair"]
    SEED = False

    ORGANS = ["abdominal_wall",
                "colon",
                "inferior_mesenteric_artery",
                "intestinal_veins",
                "liver",
                "pancreas",
                "small_intestine",
                "spleen",
                "stomach",
                "ureter",
                "vesicular_glands"]

    VAL_IDS = ["03", "21", "26"] # Validation IDs of DSAD

    TEST_IDS = ["02", "07", "11", "13", "14", "18", "20", "32"] # Test IDs of DSAD

    # Parameters for training
    NUM_CLASSES = 2
    BATCH_SIZE = 16
    MINI_BATCH_SIZE = 16
    IMAGE_SIZE = (640, 512)

    # def __init__(self, organ_id=ORGAN_ID, organs=ORGANS, 
    #              data_dir=DATA_DIR, output_dir=OUTPUT_DIR,
    #              segformer=SEGFORMER, unet=UNET,
    #              mixed_precision=MIXED_PRECISION, epochs=EPOCHS,
    #              val_ids=VAL_IDS, test_ids=TEST_IDS,):
        
        # self.organ_id = organ_id
        # self.data_dir = data_dir
        # self.output_dir = output_dir
        # self.segformer = segformer
        # self.unet = unet
        # self.mixed_precision = mixed_precision
        # self.epochs = epochs
        # self.val_ids = val_ids
        # self.test_ids = test_ids
    def __init__(self, organs=ORGANS, organ_id=ORGAN_ID,
                 batch_size=BATCH_SIZE, mini_batch_size=MINI_BATCH_SIZE,
                 unet=UNET, segformer=SEGFORMER,
                 output_dir=OUTPUT_DIR):
        
        for (key, value) in vars(Config).items():
            if not key.startswith("_"):
                setattr(self, key.lower(), value)

        self.organ = self.organs[organ_id]
        self.aug = self.augs[self.aug_id]
        self.num_mini_batches = self.batch_size//self.mini_batch_size
        
        self.output_folder = "Seg_single_" + self.organ
        if self.unet:
            self.output_folder += "_unet"

        if self.segformer:
            self.output_folder += "_segformer"
            
        self.output_folder = os.path.join(self.output_dir, self.output_folder)

        os.makedirs(self.output_folder, exist_ok=True)

        self.train_transform = A.Compose(
        [
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=5, p=0.5),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

        self.val_transform = A.Compose(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    def __str__(self) -> str:
        attrs = vars(self)
        consts = vars(Config)
        attr_str = self._get_str(attrs)
        # const_str = self._get_str(consts)
        return f"Config attributes: \n\t{attr_str}"

    def _get_str(self, var_iter):
        return ',\n\t'.join("%s: %s" % item for item in var_iter.items())      
