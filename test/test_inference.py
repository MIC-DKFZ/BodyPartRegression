import sys, os, json
import nibabel as nib

sys.path.append("../")
from bpreg.scripts.bpreg_inference import *
from bpreg.settings import *

# test if json files are craeted


def test_inference():
    """Expensive test to test based on 10 nifti files in the data/test_cases
    folder if the inference method runs and creates json files.
    """

    input_path = os.path.join(MAIN_PATH, "data/test_cases/")
    output_path = os.path.join(MAIN_PATH, "data/test_results/")

    nifti_files = len(os.listdir(input_path))

    # remove existing files in the output path
    for file in os.listdir(output_path):
        filepath = output_path + file
        if "test_results" in filepath:
            os.remove(filepath)

    # run
    bpreg_inference(input_path, output_path, plot=True)

    # Test creation of json files
    json_output_files = [f for f in os.listdir(output_path) if f.endswith(".json")]
    assert len(json_output_files) == nifti_files

    # Test tags
    with open(output_path + json_output_files[0]) as f:
        x = json.load(f)

    assert "settings" in x.keys()
    assert "cleaned slice scores" in x.keys()
    assert "unprocessed slice scores" in x.keys()
    assert "body part examined" in x.keys()
    assert "body part examined tag" in x.keys()

    # Check if readme file is saved.
    assert "README.md" in os.listdir(output_path)


def test_plot_scores_in_json_files():
    output_path = os.path.join(MAIN_PATH, "data/test_results/")
    plot_scores_in_json_files(output_path)

    assert len([f for f in os.listdir(output_path) if f.endswith(".json")]) == len(
        [f for f in os.listdir(output_path) if f.endswith(".png")]
    )


def test_npy_inference():
    """Test if npy inference is equal to nifti inference."""
    input_path = os.path.join(MAIN_PATH, "data/test_cases/")
    input_path += os.listdir(input_path)[0]

    model = InferenceModel()
    metadata1 = model.nifti2json(input_path, "")

    img = nib.load(input_path)
    pixel_spacings = img.header.get_zooms()
    X = img.get_fdata(dtype=np.float32)
    metadata2 = model.npy2json(X, "", pixel_spacings=pixel_spacings)

    assert metadata1 == metadata2


if __name__ == "__main__":
    test_plot_scores_in_json_files()
