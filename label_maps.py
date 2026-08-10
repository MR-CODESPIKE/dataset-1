"""
Per-source label mapping tables.

Each entry maps a RAW class/folder name (exactly as it appears in the
source dataset) to (domain, species, disease_name). disease_name is
already normalize_label()-friendly (snake_case English).

These tables are the part that genuinely needs a human eye per dataset --
folder-name conventions differ across every source. Fill in / correct any
gaps as you inspect the actual downloaded folder structure; the downloader
scripts fail loudly (KeyError) on unmapped labels rather than silently
mislabeling data, so nothing slips through unnoticed.
"""

# ---------------------------------------------------------------- CROP ----

PLANTVILLAGE_MAP = {
    # emmarex/plantdisease Kaggle mirror uses the classic
    # "Species___Disease" folder naming (same convention as the original
    # spMohanty repo). Full 38-class list.
    "Apple___Apple_scab": ("crop", "apple", "apple_scab"),
    "Apple___Black_rot": ("crop", "apple", "black_rot"),
    "Apple___Cedar_apple_rust": ("crop", "apple", "cedar_apple_rust"),
    "Apple___healthy": ("crop", "apple", "healthy"),
    "Blueberry___healthy": ("crop", "blueberry", "healthy"),
    "Cherry_(including_sour)___Powdery_mildew": ("crop", "cherry", "powdery_mildew"),
    "Cherry_(including_sour)___healthy": ("crop", "cherry", "healthy"),
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": ("crop", "maize", "gray_leaf_spot"),
    "Corn_(maize)___Common_rust_": ("crop", "maize", "common_rust"),
    "Corn_(maize)___Northern_Leaf_Blight": ("crop", "maize", "northern_leaf_blight"),
    "Corn_(maize)___healthy": ("crop", "maize", "healthy"),
    "Grape___Black_rot": ("crop", "grape", "black_rot"),
    "Grape___Esca_(Black_Measles)": ("crop", "grape", "esca_black_measles"),
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": ("crop", "grape", "leaf_blight"),
    "Grape___healthy": ("crop", "grape", "healthy"),
    "Orange___Haunglongbing_(Citrus_greening)": ("crop", "orange", "citrus_greening"),
    "Peach___Bacterial_spot": ("crop", "peach", "bacterial_spot"),
    "Peach___healthy": ("crop", "peach", "healthy"),
    "Pepper,_bell___Bacterial_spot": ("crop", "pepper", "bacterial_spot"),
    "Pepper,_bell___healthy": ("crop", "pepper", "healthy"),
    "Potato___Early_blight": ("crop", "potato", "early_blight"),
    "Potato___Late_blight": ("crop", "potato", "late_blight"),
    "Potato___healthy": ("crop", "potato", "healthy"),
    "Raspberry___healthy": ("crop", "raspberry", "healthy"),
    "Soybean___healthy": ("crop", "soybean", "healthy"),
    "Squash___Powdery_mildew": ("crop", "squash", "powdery_mildew"),
    "Strawberry___Leaf_scorch": ("crop", "strawberry", "leaf_scorch"),
    "Strawberry___healthy": ("crop", "strawberry", "healthy"),
    "Tomato___Bacterial_spot": ("crop", "tomato", "bacterial_spot"),
    "Tomato___Early_blight": ("crop", "tomato", "early_blight"),
    "Tomato___Late_blight": ("crop", "tomato", "late_blight"),
    "Tomato___Leaf_Mold": ("crop", "tomato", "leaf_mold"),
    "Tomato___Septoria_leaf_spot": ("crop", "tomato", "septoria_leaf_spot"),
    "Tomato___Spider_mites Two-spotted_spider_mite": ("crop", "tomato", "spider_mites"),
    "Tomato___Target_Spot": ("crop", "tomato", "target_spot"),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": ("crop", "tomato", "yellow_leaf_curl_virus"),
    "Tomato___Tomato_mosaic_virus": ("crop", "tomato", "mosaic_virus"),
    "Tomato___healthy": ("crop", "tomato", "healthy"),
}

CASSAVA_MAP = {
    # nirmalsankalana/cassava-leaf-disease-classification actual folder
    # names confirmed from a live run log
    "Cassava___bacterial_blight": ("crop", "cassava", "cassava_bacterial_blight"),
    "Cassava___brown_streak_disease": ("crop", "cassava", "cassava_brown_streak_disease"),
    "Cassava___green_mottle": ("crop", "cassava", "cassava_green_mottle"),
    "Cassava___mosaic_disease": ("crop", "cassava", "cassava_mosaic_disease"),
    "Cassava___healthy": ("crop", "cassava", "healthy"),
    # keep earlier guesses too in case of alternate releases
    "0": ("crop", "cassava", "cassava_bacterial_blight"),
    "1": ("crop", "cassava", "cassava_brown_streak_disease"),
    "2": ("crop", "cassava", "cassava_green_mottle"),
    "3": ("crop", "cassava", "cassava_mosaic_disease"),
    "4": ("crop", "cassava", "healthy"),
    "Cassava Bacterial Blight (CBB)": ("crop", "cassava", "cassava_bacterial_blight"),
    "Cassava Brown Streak Disease (CBSD)": ("crop", "cassava", "cassava_brown_streak_disease"),
    "Cassava Green Mottle (CGM)": ("crop", "cassava", "cassava_green_mottle"),
    "Cassava Mosaic Disease (CMD)": ("crop", "cassava", "cassava_mosaic_disease"),
    "Healthy": ("crop", "cassava", "healthy"),
}

MAIZE_MAP = {
    # smaranjitghose/corn-or-maize-leaf-disease-dataset folder names
    "Common_Rust": ("crop", "maize", "common_rust"),
    "Gray_Leaf_Spot": ("crop", "maize", "gray_leaf_spot"),
    "Blight": ("crop", "maize", "blight"),
    "Healthy": ("crop", "maize", "healthy"),
}

# TOM2024 label mapping is intentionally left to be filled in after first
# download -- Mendeley zips typically ship as folder-per-class but exact
# class folder names for TOM2024's 30 classes weren't verifiable before
# download. The downloader script will print all unmapped folder names
# it finds so you can extend TOM2024_MAP in one pass.
TOM2024_MAP = {
    "Healthy_leaf_maize": ("crop", "maize", "healthy"),
    "Rust": ("crop", "maize", "common_rust"),
    "Curvularia": ("crop", "maize", "curvularia_leaf_spot"),
    "Helminthosporiosis": ("crop", "maize", "helminthosporiosis"),
    "Virose_maize": ("crop", "maize", "viral_disease"),
    "Stripe": ("crop", "maize", "maize_streak_virus"),
    "Abiotic_disease": ("crop", "maize", "abiotic_stress"),
    "Fall_Armyworm_Activity": ("crop", "maize", "fall_armyworm"),
    "Armyworm": ("crop", "maize", "armyworm"),
    "Aphids": ("crop", "maize", "aphids"),
    "Healthy_leaf_onion": ("crop", "onion", "healthy"),
    "Fusarium": ("crop", "onion", "fusarium"),
    "Alternaria": ("crop", "onion", "alternaria"),
    "Virosis_onion": ("crop", "onion", "viral_disease"),
    "Bulb_rot": ("crop", "onion", "bulb_rot"),
    "Caterpillars": ("crop", "onion", "caterpillars"),
}

# --------------------------------------------------------------- ANIMAL ---

CATTLE_LSD_MAP = {
    # shivamagarwal29/cow-lumpy-disease-dataset actual folder names
    # (confirmed from a live run log)
    "lumpycows": ("animal", "cattle", "lumpy_skin_disease"),
    "healthycows": ("animal", "cattle", "healthy"),
    # keep the earlier guesses too in case a re-upload changes casing/format
    "Lumpy Skin": ("animal", "cattle", "lumpy_skin_disease"),
    "Normal Skin": ("animal", "cattle", "healthy"),
    "Lumpy": ("animal", "cattle", "lumpy_skin_disease"),
    "Normal": ("animal", "cattle", "healthy"),
}

POULTRY_MAP = {
    # allandclive/chicken-disease-1 -- confirmed from a live run: images
    # sit flat inside a single Train/ folder with class encoded in the
    # filename prefix (e.g. "salmo.1600.jpg", "healthy.1929.jpg",
    # "pcrcocci.291.jpg"). Handled via find_classes_by_filename_prefix().
    "salmo": ("animal", "poultry", "salmonella"),
    "healthy": ("animal", "poultry", "healthy"),
    "pcrcocci": ("animal", "poultry", "coccidiosis"),
    "cocci": ("animal", "poultry", "coccidiosis"),
    "ncd": ("animal", "poultry", "newcastle_disease"),
    "pcrncd": ("animal", "poultry", "newcastle_disease"),
    "pcrsalmo": ("animal", "poultry", "salmonella"),
    # keep folder-name-style guesses too in case of alternate releases
    "Coccidiosis": ("animal", "poultry", "coccidiosis"),
    "Healthy": ("animal", "poultry", "healthy"),
    "New Castle Disease": ("animal", "poultry", "newcastle_disease"),
    "Salmonella": ("animal", "poultry", "salmonella"),
}

# ---------------------------------------------------------------- HUMAN ---

# DermNet's 23 top-level classes (shubhamgoel27/dermnet) -- folder names
# from the standard Kaggle release.
DERMNET_MAP = {
    "Acne and Rosacea Photos": ("human", "skin", "acne_rosacea"),
    "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions": ("human", "skin", "malignant_lesion"),
    "Atopic Dermatitis Photos": ("human", "skin", "atopic_dermatitis"),
    "Bullous Disease Photos": ("human", "skin", "bullous_disease"),
    "Cellulitis Impetigo and other Bacterial Infections": ("human", "skin", "bacterial_infection"),
    "Eczema Photos": ("human", "skin", "eczema"),
    "Exanthems and Drug Eruptions": ("human", "skin", "drug_eruption"),
    "Hair Loss Photos Alopecia and other Hair Diseases": ("human", "skin", "alopecia"),
    "Herpes HPV and other STDs Photos": ("human", "skin", "std_related"),
    "Light Diseases and Disorders of Pigmentation": ("human", "skin", "pigmentation_disorder"),
    "Lupus and other Connective Tissue diseases": ("human", "skin", "connective_tissue_disease"),
    "Melanoma Skin Cancer Nevi and Moles": ("human", "skin", "melanoma"),
    "Nail Fungus and other Nail Disease": ("human", "skin", "nail_fungus"),
    "Poison Ivy Photos and other Contact Dermatitis": ("human", "skin", "contact_dermatitis"),
    "Psoriasis pictures Lichen Planus and related diseases": ("human", "skin", "psoriasis"),
    "Scabies Lyme Disease and other Infestations and Bites": ("human", "skin", "scabies_infestation"),
    "Seborrheic Keratoses and other Benign Tumors": ("human", "skin", "seborrheic_keratosis"),
    "Systemic Disease": ("human", "skin", "systemic_disease"),
    "Tinea Ringworm Candidiasis and other Fungal Infections": ("human", "skin", "fungal_infection"),
    "Urticaria Hives": ("human", "skin", "urticaria"),
    "Vascular Tumors": ("human", "skin", "vascular_tumor"),
    "Vasculitis Photos": ("human", "skin", "vasculitis"),
    "Warts Molluscum and other Viral Infections": ("human", "skin", "viral_infection"),
}

# SkinCAP (joshuachou/SkinCAP on HF) ships free-text disease labels per row
# rather than fixed folders -- the downloader normalizes these dynamically
# via normalize_label() rather than a fixed lookup table, so no map needed
# here. See download_hf_dataset.py --dataset skincap for handling.
SKINCAP_DYNAMIC = True
