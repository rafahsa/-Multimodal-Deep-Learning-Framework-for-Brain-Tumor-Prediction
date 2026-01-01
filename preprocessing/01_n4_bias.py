# %%
"""
Step 01: N4 Bias Field Correction
BraTS 2018 - MRI Preprocessing
"""

import os
import SimpleITK as sitk
import cupy as cp
from tqdm import tqdm

# =========================
# Paths
# =========================
RAW_DIR = "data/brats2018/MICCAI_BraTS_2018_Data_Training"
OUT_DIR = "data/processed_n4"

CLASSES = ["HGG", "LGG"]
MODALITIES = ["flair", "t1", "t1ce", "t2"]

os.makedirs(OUT_DIR, exist_ok=True)

def apply_n4(input_nii, output_nii):
    # قراءة الصورة باستخدام SimpleITK
    image = sitk.ReadImage(input_nii, sitk.sitkFloat32)

    # تحويل الصورة إلى numpy array ثم إلى GPU باستخدام CuPy
    image_data = cp.asarray(sitk.GetArrayFromImage(image))

    # إنشاء القناع باستخدام Otsu thresholding (نقل العمليات الحسابية إلى GPU)
    # التعديل هنا باستخدام CuPy لجعل العملية أسرع على GPU
    mask = cp.where(image_data > cp.percentile(image_data, 50), 1, 0)

    # تطبيق تصحيح الإضاءة N4ITK باستخدام SimpleITK (لا يمكن استبداله بـ CuPy)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()

    # تسريع المعالجة باستخدام 8 threads
    corrector.SetMaximumNumberOfIterations([20, 20])
    corrector.SetConvergenceThreshold(1e-3)
    corrector.SetNumberOfThreads(8)

    corrected = corrector.Execute(image, sitk.GetImageFromArray(mask.get()))  # تحويل mask من CuPy إلى SimpleITK
    sitk.WriteImage(corrected, output_nii)

# =========================
# Main Loop
# =========================
for cls in CLASSES:
    in_cls_dir = os.path.join(RAW_DIR, cls)
    out_cls_dir = os.path.join(OUT_DIR, cls)
    os.makedirs(out_cls_dir, exist_ok=True)

    patients = sorted(os.listdir(in_cls_dir))
    print(f"\nProcessing {cls} ({len(patients)} patients)")

    for patient in tqdm(patients):
        in_patient_dir = os.path.join(in_cls_dir, patient)
        out_patient_dir = os.path.join(out_cls_dir, patient)
        os.makedirs(out_patient_dir, exist_ok=True)

        for mod in MODALITIES:
            nii_name = f"{patient}_{mod}.nii"   # ✅ الصحيح
            in_nii = os.path.join(in_patient_dir, nii_name)
            out_nii = os.path.join(out_patient_dir, nii_name)

            if not os.path.exists(in_nii):
                print(f"❌ Missing: {in_nii}")
                continue

            apply_n4(in_nii, out_nii)

print("\n✅ N4 Bias Field Correction completed successfully using GPU-accelerated CuPy!")
