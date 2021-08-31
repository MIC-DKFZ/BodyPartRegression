"""
Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0
   
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import pandas as pd
import pydicom
import argparse
import glob
import os
import SimpleITK as sitk


class DicomHandler:
    def __init__(self):
        pass

    def base_df(self, path, study, suffix=""):
        df = self.dir2df(path)
        df = df.drop_duplicates("base_path").reset_index(drop=True)
        df.index = df["base_path"]

        df["dirname"] = df["base_path"].apply(
            lambda x: suffix + x.replace(path, "").replace("/", "__").replace(" ", "--")
        )
        df["broken_path"] = 0
        df["id_name"] = [study + "_" + i for i in np.arange(len(df)).astype(str)]
        df["success"] = 0
        df.drop(["file_path", "file_name"], axis=1, inplace=True)

        return df

    def metaData2Series(self, path):
        """
        Input: pydicom Dataset
        Output: Pandas Series
        """

        def func(x):
            if isinstance(x, pydicom.multival.MultiValue):
                x = str(x)
            return x

        ds = pydicom.dcmread(path)
        dcmDict = {key: func(ds[key].value) for key in ds.dir() if key != "PixelData"}
        series = pd.Series(dcmDict)

        return series

    def dir2df(self, path):
        file_paths = glob.glob(path + "/**/*.dcm", recursive=True)
        file_names = [f.split("/")[-1] for f in file_paths]
        base_paths = [
            file_paths[i].replace(file_names[i], "") for i in range(len(file_paths))
        ]

        df = pd.DataFrame(index=range(0, len(file_paths)))
        df["file_path"] = file_paths
        df["file_name"] = file_names
        df["base_path"] = base_paths
        df["images_in_dir"] = df["base_path"].map(
            df.groupby("base_path")["file_name"].count()
        )

        return df

    def add_unknown_columns(self, df, series):
        colsNotInDf = list(set(series.keys()) - set(df.columns))
        for col in colsNotInDf:
            df[col] = np.nan

    def dicom2nifti(self, ifilepath, ofilepath):
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(ifilepath)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        sitk.WriteImage(image, ofilepath)

    def dicom2array(self, ifilepath):
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(ifilepath)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        array = sitk.GetArrayFromImage(image)
        return array

    def nrrd2nifti(self, ifilepath, ofilepath):
        reader = sitk.ImageFileReader()
        reader.SetImageIO("NrrdImageIO")
        reader.SetFileName(ifilepath)
        image = reader.Execute()
        sitk.WriteImage(image, ofilepath)

    def get_first_image(self, x, case="default"):

        if case == "CQ500":
            return "CT000001.dcm"

        filename = "1-1.dcm"
        for i, y in enumerate([10, 100, 1000, 10000], 1):
            if x / y >= 1:
                filename = "1-" + "0" * i + "1.dcm"

        return filename


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", default="")
    parser.add_argument("-o", default="")

    value = parser.parse_args()
    input_dirpath = value.i
    output_dirpath = value.o

    dh = DicomHandler()
    for file in os.listdir(input_dirpath):
        print(file)
        ifilepath = os.path.join(input_dirpath, file)
        ofilepath = os.path.join(output_dirpath, file.replace(".dcm", ".nii.gz"))
        dh.dicom2nifti(ifilepath, ofilepath)
