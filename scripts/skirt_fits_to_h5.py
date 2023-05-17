import h5py
import os
from glob import glob
from astropy.io import fits
from tqdm import tqdm
import numpy as np


def main(args):
    files = []
    for directory in tqdm(os.listdir(args.skirt_path)):
        for f in glob(os.path.join(args.skirt_path, directory, "*.fits")):
            files.append(f)
    with h5py.File(args.output_path, "w") as hf:
        dt = h5py.string_dtype(encoding='utf-8') # allows storing variable length strings
        hf.create_dataset("images", [len(files), len(args.filters), args.size, args.size], dtype=np.float32, chunks=(1, 1, 512, 512))
        hf["images"].attrs["units"] = 'AB mag/arcsec2'
        for i, _filter in enumerate(args.filters):
            hf["images"].attrs[f"channel_{i}"] = _filter
        hf.create_dataset("SIMTAG",  [len(files)],  dtype=dt)
        hf["SIMTAG"].attrs["description"] = "Simulation name"
        hf.create_dataset("SNAPNUM", [len(files)],  dtype="i8")
        hf["SNAPNUM"].attrs["description"] = "Simulation snapshot"
        hf.create_dataset("SUBHALO", [len(files)],  dtype="i8")
        hf["SUBHALO"].attrs["description"] = "Subhalo ID"
        hf.create_dataset("INCL",    [len(files)],  dtype="f")
        hf["INCL"].attrs["description"] = "Camera inclination"
        hf["INCL"].attrs["units"] = "deg"
        hf.create_dataset("AZIM",    [len(files)],  dtype="f")
        hf["AZIM"].attrs["description"] = "Camera azimuth"
        hf["AZIM"].attrs["units"] = "deg"
        hf.create_dataset("ROLL",    [len(files)],  dtype="f")
        hf["ROLL"].attrs["description"] = "Camera roll"
        hf["ROLL"].attrs["units"] = "deg"
        hf.create_dataset("REDSHIFT",[len(files)],  dtype="f")
        hf.create_dataset("FOVSIZE", [len(files)],  dtype="f")
        hf["FOVSIZE"].attrs["description"] = "Field of view of the scene in kpc"
        hf["FOVSIZE"].attrs["units"] = "kpc"
        hf.create_dataset("CDELT1",  [len(files)],  dtype="f")
        hf["CDELT1"].attrs["description"] = "Coordinate increment along the X-axis in kpc"
        hf["CDELT1"].attrs["units"] = "kpc"
        hf.create_dataset("CDELT2",  [len(files)],  dtype="f")
        hf["CDELT2"].attrs["description"] = "Coordinate increment along the Y-axis in kpc"
        hf["CDELT2"].attrs["units"] = "kpc"
        hf.create_dataset("filename", [len(files)], dtype=dt)
        hf["filename"].attrs["description"] = "Filename of the fits file that the image belongs to"

        for i, f in (pbar := tqdm(enumerate(files))):
            data = fits.open(f)
            filename = os.path.split(f)[-1]
            pbar.set_description(filename)
            hdr = data[0].header
            total_size = hdr["NAXIS1"]
            if total_size >= args.size:
                left_crop = (total_size - args.size) // 2
                right_crop = left_crop + ((total_size - args.size) % 2)
                for j, _filter in enumerate(args.filters):
                    image = data[_filter].data
                    hf["images"][i, j] = image[left_crop:total_size - right_crop, left_crop:total_size - right_crop],
            else:
                left_pad = (args.size - total_size) // 2
                right_pad = left_pad + ((args.size - total_size) % 2)
                for j, _filter in enumerate(args.filters):
                    image = data[_filter].data
                    hf["images"][i, j] = np.pad(image, [[left_pad, right_pad]]*2, mode="constant", constant_values=99),
            hf["SIMTAG"][i] = hdr["SIMTAG"]
            hf["SNAPNUM"][i] = hdr["SNAPNUM"]
            hf["SUBHALO"][i] = hdr["SUBHALO"]
            hf["INCL"][i] = hdr["INCL"]
            hf["AZIM"][i] = hdr["AZIM"]
            hf["ROLL"][i] = hdr["ROLL"]
            hf["FOVSIZE"][i] = args.size * hdr["CDELT1"]
            hf["CDELT1"][i] = hdr["CDELT1"]
            hf["CDELT2"][i] = hdr["CDELT2"]
            hf["filename"][i] = filename


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--size", default=512, type=int)
    parser.add_argument("--filters", nargs="+", required=True)
    parser.add_argument("--skirt_path", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()
    main(args)
