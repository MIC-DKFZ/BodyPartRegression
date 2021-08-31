import numpy as np
import pandas as pd
import SimpleITK as sitk
import os, sys
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from ipywidgets import widgets, interactive

sys.path.append("../../")
from bpreg.preprocessing.nifti2npy import *
from bpreg.settings import *
from bpreg.inference.inference_model import InferenceModel


def dicom2nifti(ifilepath, ofilepath):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(ifilepath)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    sitk.WriteImage(image, ofilepath)


def convert_ct_lymph_nodes_to_nifti(dicom_path, nifti_path):
    def get_ct_lymph_node_dicom_dir(base_path):
        path = base_path + "/" + os.listdir(base_path)[0] + "/"
        path += os.listdir(path)[0] + "/"
        return path

    dicom_dirs = [
        mydir
        for mydir in np.sort(os.listdir(dicom_path))
        if mydir.startswith(("ABD", "MED"))
    ]

    for dicom_dir in tqdm(dicom_dirs):
        ifilepath = get_ct_lymph_node_dicom_dir(dicom_path + dicom_dir)
        ofilepath = nifti_path + dicom_dir + ".nii.gz"
        if os.path.exists(ofilepath):
            continue
        dicom2nifti(ifilepath, ofilepath)


def nifti2npy(nifti_path, npy_path):
    base_path = "/".join(nifti_path.split("/")[0:-2]) + "/"
    n2n = Nifti2Npy(
        target_pixel_spacing=7,  # in mm/pixel
        min_hu=-1000,  # in Hounsfield units
        max_hu=1500,  # in Hounsfield units
        size=64,  # x/y size
        ipath=nifti_path,  # input path
        opath=npy_path,  # output path
        rescale_max=1,  # rescale max value
        rescale_min=-1,
    )  # rescale min value
    filepaths = np.sort([nifti_path + f for f in os.listdir(nifti_path)])

    df = n2n.convert(filepaths, save=True)
    df.to_excel(base_path + "meta_data.xlsx")


def update_meta_data(landmark_filepath, meta_data_filepath):
    """
    add information to train, val and test data to dataframe
    """
    df_landmarks = pd.read_excel(landmark_filepath, sheet_name="database")
    df_meta_data = pd.read_excel(meta_data_filepath, index_col=0)

    train_filenames = [
        f + ".npy" for f in df_landmarks.loc[df_landmarks.train == 1, "filename"]
    ]
    val_filenames = [
        f + ".npy" for f in df_landmarks.loc[df_landmarks.val == 1, "filename"]
    ]
    test_filenames = [
        f + ".npy" for f in df_landmarks.loc[df_landmarks.test == 1, "filename"]
    ]

    df_meta_data["train_data"] = 1
    df_meta_data.loc[val_filenames, "val_data"] = 1
    df_meta_data.loc[test_filenames, "test_data"] = 1
    df_meta_data.loc[val_filenames + test_filenames, "train_data"] = 0

    df_meta_data.to_excel(meta_data_filepath)


def preprocess_ct_lymph_node_dataset(dicom_path, nifti_path, npy_path):
    """Convert DICOM files form CT Lymph node to downsampled npy volumes."""
    # Convert Dicom to nifti
    convert_ct_lymph_nodes_to_nifti(dicom_path, nifti_path)

    # Convert nifti files to npy and save meta_data.xlsx file
    nifti2npy(nifti_path, npy_path)

    # update meta data with train/val/test data from landmark file
    update_meta_data(
        landmark_filepath="data/ct-lymph-nodes-annotated-landmarks.xlsx",
        meta_data_filepath="data/meta_data.xlsx",
    )


################## Inference notebook


def crop_scores(scores, start_score, end_score):
    scores = np.array(scores)
    min_scores = np.where(scores < start_score)[0]
    max_scores = np.where(scores > end_score)[0]

    min_index = 0
    max_index = len(scores)

    if len(min_scores) > 0:
        min_index = np.nanmax(min_scores)

    if len(max_scores) > 0:
        max_index = np.nanmin(max_scores)

    return min_index, max_index


def plot_dicomexamined_distribution(
    df, column="tag", count_column="count", fontsize=20
):

    color_counts = len(np.unique(df[column])) + 2
    cmap = plt.cm.get_cmap("cubehelix", color_counts)
    colors = [cmap(i) for i in range(color_counts - 1)]

    bodypartexamined_dicomtags = df.groupby(column)[count_column].count() / len(df)

    if bodypartexamined_dicomtags.sum() != 1:
        bodypartexamined_dicomtags["NONE"] = 1 - bodypartexamined_dicomtags.sum()

    bodypartexamined_dicomtags = bodypartexamined_dicomtags.sort_values()

    _, ax = plt.subplots(figsize=(10, 10))
    _, texts, autotexts = ax.pie(
        bodypartexamined_dicomtags.values * 100,
        labels=bodypartexamined_dicomtags.index,
        autopct="%1.1f%%",
        colors=colors,
    )

    for i, txt, txt2 in zip(np.arange(len(texts)), texts, autotexts):
        txt.set_fontsize(fontsize)
        txt2.set_fontsize(fontsize - 2)
        if i < 4:
            txt2.set_color("white")

    ax.axis("equal")
    plt.tight_layout()

    bodypartexamined_dicomtags = bodypartexamined_dicomtags.sort_values(ascending=False)
    bodypartexamined_dicomtags = pd.DataFrame(
        np.round(bodypartexamined_dicomtags * 100, 1)
    )
    bodypartexamined_dicomtags.columns = ["Proportion [%]"]
    return bodypartexamined_dicomtags


def load_json(filepath):
    with open(filepath) as f:
        x = json.load(f)
    return x


def plot_scores(filepath, fontsize=18):
    plt.figure(figsize=(12, 6))
    x = load_json(filepath)

    plt.plot(x["z"], x["cleaned slice scores"], label="cleaned slice scores")
    plt.plot(
        x["z"],
        x["unprocessed slice scores"],
        label="unprocessed slice scores",
        linestyle="--",
    )

    try:
        min_score = np.nanmin(x["unprocessed slice scores"])
        max_score = np.nanmax(x["unprocessed slice scores"])
        dflandmarks = pd.DataFrame(x["look-up table"]).T
        landmarks = dflandmarks[
            (dflandmarks["mean"] > min_score) & (dflandmarks["mean"] < max_score)
        ]

        for landmark, row in landmarks.iloc[[0, -1]].iterrows():
            plt.plot(
                [0, np.nanmax(x["z"])],
                [row["mean"], row["mean"]],
                linestyle=":",
                color="black",
                linewidth=0.8,
            )
            plt.text(
                5,
                row["mean"] + 1,
                landmark,
                fontsize=fontsize - 4,
                bbox=dict(
                    boxstyle="square",
                    fc=(1.0, 1, 1),
                ),
            )
    except:
        pass

    plt.xlabel("height [mm]", fontsize=fontsize)
    plt.ylabel("Slice Scores", fontsize=fontsize)
    plt.xticks(fontsize=fontsize - 2)
    plt.yticks(fontsize=fontsize - 2)
    plt.legend(loc=1, fontsize=fontsize)
    plt.xlim((0, np.nanmax(x["z"])))

    filename = filepath.split("/")[-1]
    if len(filename) > 60:
        plt.title(f"{filename[:60]}...\n", fontsize=fontsize - 2)
    else:
        plt.title(filename + "\n", fontsize=fontsize - 2)
    plt.show()


def get_updated_bodypartexamined_from_json_files(data_path):
    files = [f for f in os.listdir(data_path) if f.endswith(".json")]
    dftags = pd.DataFrame(index=files, columns=["tag"])
    for file in files:
        with open(os.path.join(data_path, file), "rb") as f:
            x = json.load(f)
        dftags.loc[file, "tag"] = x["body part examined tag"]
    dftags["count"] = 1
    return dftags


# Interactive plot to visualize scores
def plot_scores_interactive(json_filepaths, nifti_filepaths):
    def plotit(idx):
        filepath = json_filepaths[int(idx)]
        plot_scores(filepath)
        plot_volume(nifti_filepaths[int(idx)])

    idx = widgets.BoundedFloatText(
        value=0,
        min=0,
        max=len(json_filepaths) - 1,
        step=1,
        description="JSON file:",
        disabled=False,
        color="black",
    )

    return interactive(plotit, idx=idx)


def plot_volume(filepath, min_index=0, max_index=np.nan, title=""):
    X, pixel_spacings = load_nifti_volume(filepath.replace(".json", ".nii.gz"))
    plt.figure(figsize=(6, 6))
    if np.isnan(max_index):
        max_index = X.shape[2]
    plt.imshow(
        X[:, 250, min_index:max_index].T,
        origin="lower",
        cmap="gray",
        vmax=1500,
        vmin=-1000,
        aspect="auto",
    )
    plt.xticks([])
    plt.yticks([])
    if len(title) > 0:
        plt.title(title, fontsize=18)
    plt.show()

    return X


# Interactive plot to visualize tailerd volumes
def plot_tailored_volumes_interactive(
    dfchests, start_score, end_score, json_base_path, nifti_base_path
):
    def plotit(idx):
        filename = dfchests.loc[idx, "json"]  # 0, 10, 13, 14
        x = load_json(os.path.join(json_base_path, filename))
        min_index, max_index = crop_scores(
            x["cleaned slice scores"], start_score, end_score
        )

        _ = plot_volume(
            os.path.join(nifti_base_path, filename), title="Original Volume"
        )
        _ = plot_volume(
            os.path.join(nifti_base_path, filename),
            min_index,
            max_index,
            title="Tailored Volume",
        )

    idx = widgets.BoundedFloatText(
        value=0,
        min=0,
        max=len(dfchests) - 1,
        step=1,
        description="JSON file:",
        disabled=False,
        color="black",
    )
    return interactive(plotit, idx=idx)
