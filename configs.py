"""
Configuration file for data paths
Set USE_HPC=True when running on HPC cluster, False for local development
"""
from pathlib import Path

# Toggle between HPC and local paths
USE_HPC = True

if USE_HPC:
    # HPC paths
    DATA_DIR = Path("/gpfs/data/lattanzilab/Ilias/NYU_UCSF_Collab")
    DICOM_NORMAL = DATA_DIR / "Dicom_Normal"
    DICOM_MENISCAL = DATA_DIR / "Dicom_Meniscal_Tear"

    # Annotation paths
    ANNOTATIONS_DIR = DATA_DIR / "Annotations"
    MENISCUS_ANNOTATIONS_DIR = ANNOTATIONS_DIR / "meniscus_specific"
    NORMAL_ANNOTATIONS_DIR = ANNOTATIONS_DIR / "normal_no_anomalies"

    # CSV files
    MENISCUS_MDAI_CSV = MENISCUS_ANNOTATIONS_DIR / "TBRecon_anomaly_meniscus_MDai_df.csv"
    MENISCUS_ID_KEY_CSV = MENISCUS_ANNOTATIONS_DIR / "TBRecon_ID_key_meniscus.csv"
    NORMAL_MDAI_CSV = NORMAL_ANNOTATIONS_DIR / "TBRecon_anomaly_normal_MDai_df.csv"
    NORMAL_ID_KEY_CSV = NORMAL_ANNOTATIONS_DIR / "TBRecon_ID_key_normal.csv"

else:
    # Local development paths
    DATA_DIR = Path("data")
    DICOM_NORMAL = DATA_DIR / "dicoms"
    DICOM_MENISCAL = DATA_DIR / "dicoms"

    # For local development, CSVs are in the same directory
    MENISCUS_ANNOTATIONS_DIR = DATA_DIR / "dicoms"
    NORMAL_ANNOTATIONS_DIR = DATA_DIR / "dicoms"

    MENISCUS_MDAI_CSV = MENISCUS_ANNOTATIONS_DIR / "TBRecon_anomaly_meniscus_MDai_df.csv"
    MENISCUS_ID_KEY_CSV = MENISCUS_ANNOTATIONS_DIR / "TBRecon_ID_key_meniscus.csv"
    NORMAL_MDAI_CSV = NORMAL_ANNOTATIONS_DIR / "TBRecon_anomaly_meniscus_MDai_df.csv"
    NORMAL_ID_KEY_CSV = NORMAL_ANNOTATIONS_DIR / "TBRecon_ID_key_meniscus.csv"

# Output paths
OUTPUT_DIR = Path("yolo_dataset")
