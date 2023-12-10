#--------------------------------------------------------------------------
# Software:     Panoramic generator from CT

# Comments:     This is code is from paper: "Reconstruction of Panoramic
#               Dental Images Through Bézier Function Optimization"
#               https://doi.org/10.3389/fbioe.2020.00794
#
#               Code modified from original repo of @PauloJamorin

# Copyright:    (C) 2019 - CTI Renato Archer

# Authors:      Paulo H. J. Amorim (paulo.amorim (at) cti.gov.br) 
#               Thiago F. Moraes (thiago.moraes (at) cti.gov.br)
#               Jorge V. L. Silva (jorge.silva (at) cti.gov.br)
#               Helio Pedrini (helio (at) ic.unicamp.br)
#               Rui B. Ruben (rui.ruben (at) ipleiria.pt)

# License:      GNU - GPL 2 (LICENSE.txt/LICENCA.txt)
#---------------------------------------------------------------------------

#This program is free software; you can redistribute it and/or
#modify it under the terms of the GNU General Public License
#as published by the Free Software Foundation; either version 2
#of the License, or (at your option) any later version.

#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.

#You should have received a copy of the GNU General Public License
#along with this program; if not, write to the Free Software
#Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
#---------------------------------------------------------------------------

import optparse as op
import pathlib
import os

import imageio
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.optimize import minimize, curve_fit

import bezier
import draw_bezier
import skeleton as skeleton

import nrrd

def open_image(filename):
    with h5py.File(filename, "r") as f:
        return f["image"][()], f["spacing"][()]

def open_nrrd_image(filename):
    input_data, input_header = nrrd.read(filename)
    return np.transpose(input_data, [2,0,1]), input_header


def save_image(image, filename, spacing=(1.0, 1.0, 1.0)):
    image_nifti = nib.Nifti1Image(np.swapaxes(np.fliplr(image), 0, 2), None)
    image_nifti.header.set_zooms(spacing)
    image_nifti.header.set_dim_info(slice=0)
    nib.save(image_nifti, filename)

def normalize_(input_data, factor):
    data_min = input_data.min()
    data_max = input_data.max()
    return (factor * (input_data-data_min))/(data_max-data_min)

def vrint(a,b,c):
    if c:
        print(a,b)
def diff_curves(control_points, skeleton_points):
    skx = skeleton_points[::2]
    sky = skeleton_points[1::2]
    bx, by = bezier.calc_bezier_curve(control_points, skx.shape[0])
    diff = ((bx - skx) ** 2 + (by - sky) ** 2).sum() ** 0.5
    return diff

def parse_comand_line():
    """
    Handle command line arguments.
    """
    usage = "usage: %prog [options] file.nrrd"  #file.hdf5"
    parser = op.OptionParser(usage)

    # -d or --debug: print all pubsub messagessent
    parser.add_option(
        "-o", "--output", help="Output file (nrdd)", default="panoramic.nrdd"
    )
    parser.add_option(
        "-d",
        "--distance",
        type="int",
        dest="distance",
        default=3,
        help="Distance between the curves",
    )
    parser.add_option(
        "-n",
        "--ncurves",
        type="int",
        dest="ncurves",
        default=10,
        help="Number of curves between and after the found curve",
    )
    parser.add_option(
        "-p",
        "--npoints",
        type="int",
        dest="npoints",
        default=500,
        help="Number of points (pixels) for each curve",
    )
    parser.add_option(
        "-g",
        "--nctrl_points",
        type="int",
        dest="nctrl_points",
        default=10,
        help="Number of bezier control points",
    )
    parser.add_option(
        "-t",
        "--threshold",
        type="int",
        dest="threshold",
        default=1500,
        help="Threshold used to determine the dental arcade",
    )
    parser.add_option(
        "-s",
        "--skeleton",
        dest="gen_skeleton",
        action="store_true",
        help="Generate skeleton image",
    )

    parser.add_option(
        "-k",
        "--skip",
        dest="skip_detect",
        action="store_true",
        help="Assume binary 3D as input (teeth)",
    )

    parser.add_option(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Verobose ,ode",
    )

    parser.add_option(
        "-l",
        "--plot",
        dest="plots",
        action="store_true",
        help="Interactive plots of intermediate results ",
    )

    options, args = parser.parse_args()

    if len(args) != 1:
        parser.error("Incorrect number of arguments")

    filename = args[0]
    return filename, options


def main():
    filename, options = parse_comand_line()
    vrint('options', options, options.verbose)

    distance = options.distance
    ncurves = options.ncurves
    npoints = options.npoints
    nctrl_points = options.nctrl_points
    threshold = options.threshold
    skip_detection = options.skip_detect
    verbose = options.verbose
    all_plots = options.plots

    output_filename = pathlib.Path(options.output).resolve()
    gen_skeleton = options.gen_skeleton

    if gen_skeleton:
        output_filename_skeleton = output_filename.parent.joinpath(
            output_filename.stem + "_skeleton" + output_filename.suffix
        )
        vrint('',output_filename_skeleton, verbose)

    # STEP 1: read image (image orientation: z,x,y)
    vrint('input file', filename, verbose)
    image, header = open_nrrd_image(filename)
    vrint('image:',[image.shape, image.min(), image.max()], verbose)

    output_filename_panoramic = os.path.splitext(os.path.basename(filename))[0]

    if all_plots:
        fig, axs = plt.subplots(3,3)
        axs[0, 0].imshow(image.max(0))
        axs[0, 1].imshow(image.max(1))
        axs[0, 2].imshow(image.max(2))

    # STEP 2: Create an skeleton in image domain
    # i) select slice st (image>threshold).sum([1,2]).argmax()), ii) apply math morpho and skeletization to slice
    if skip_detection:
        threshold = (image.max()/2)  # assumes input image is already binary
    skeleton_image, best_slice, slice_number, slice_morpho = skeleton.find_dental_arcade(image, threshold)

    if all_plots:
        axs[1, 0].imshow(slice_morpho)
        axs[1, 0].set_title('binary slice')

        axs[1, 1].imshow(skeleton_image)
        axs[1, 1].set_title('skeleton image')

    # STEP 3: extract list of 2D chain points from skeleton image
    # TODO: skeleton.im2points should be reimplemented
    a = np.where(skeleton_image>0)
    #skeleton_points = np.array(skeleton.img2points(skeleton_image), dtype=np.float64)
    skeleton_points = np.array(a).transpose()

    # set the number of chain points to npoints
    sky, skx = skeleton.normalize_curve(skeleton_points, npoints) # 500 points

    if all_plots:
        axs[1,2].imshow(image[slice_number], cmap="gray")
        axs[1,2].plot(skx, sky, '.')
        axs[1,2].set_title("slice image and skeleton normalised points")

    # STEP 4: estimate bezier curves
    # initialize, normalize then optimize
    skx_min , skx_max = skx.min() , skx.max()
    sky_min, sky_max = sky.min() , sky.max()
    opt_skeleton_points = np.empty(shape=(npoints * 2), dtype=np.float64)
    opt_skeleton_points[::2] = (skx - skx_min) / (skx_max - skx_min)
    opt_skeleton_points[1::2] = (sky - sky_min) / (sky_max - sky_min)

    initial_points = np.random.random(nctrl_points * 2)

    # estimate bezier parameters that minimize error diff_curves, given ground truth opt_ and initialisation initial
    res = minimize(
        diff_curves, initial_points, args=(opt_skeleton_points), method="SLSQP"
    )

    if not res.success:
        print("curves approximation failed")
        return

    # renormalise points
    res.x[::2] = res.x[::2] * (skx_max - skx_min) + skx_min
    res.x[1::2] = res.x[1::2] * (sky_max - sky_min) + sky_min

    # compute points along reference curve
    bx, by = bezier.calc_bezier_curve(res.x, npoints)

    # based on the reference bezier curve, estimate ncurves tangencial
    curves = (
        bezier.calc_parallel_bezier_curves(
            res.x, distance=-distance, ncurves=ncurves, npoints=npoints
        )[::-1]
        + [(bx, by)]
        +
        bezier.calc_parallel_bezier_curves(
            res.x, distance=distance, ncurves=ncurves, npoints=npoints
        )
    )
    if all_plots:
        axs[2,0].imshow(image[slice_number], cmap="gray")
        for n, curve in enumerate(curves):
            px, py = curve
            axs[2,0].plot(px, py)
        axs[2,0].plot(bx, by, '+')
        axs[2,0].set_title("bezier curves")

    # STEP 5: estimate 2D projection
    panoramic_image = draw_bezier.planify_curves(image, np.array(curves))
    vrint("panoramic computation completed", panoramic_image.shape, verbose)

    if all_plots:
        axs[2,1].imshow(panoramic_image.max(0), cmap="gray")
        axs[2,1].set_title("panoramic image")
        axs[2,2].set_axis_off()

        vrint("filename output", output_filename_panoramic, verbose)
        plt.savefig(f'{output_filename_panoramic}_fig_panoramic.png')
        plt.show()

    panoramic_image_proj = panoramic_image.max(0)
    imageio.imsave(f"{output_filename_panoramic}_panoramic.png", (normalize_(panoramic_image_proj, 255).astype(np.uint8) ))
    vrint("panoramic image saved in", f"{output_filename_panoramic}_panoramic.png", verbose)

    # save the 3d panoramic
    spacing = (1, 1, 1)
    sx, sy, sz = spacing
    sx = (
        ((sx * bx[::2] - sx * bx[1::2]) ** 2 + (sy * by[::2] - sy * by[1::2]) ** 2)
        ** 0.5
    ).mean()
    #save_image(panoramic_image, str(output_filename), spacing=(sx, sz, distance))

    # a posteriori re-create skeleton curves (?)
    if gen_skeleton:
        vrint('skeleton', '', verbose)
        sky, skx = skeleton.normalize_curve(skeleton_points, npoints)
        skeleton_curves = (
            skeleton.calc_parallel_curves(
                skx, sky, distance=-distance, ncurves=ncurves
            )[::-1]
            + [(skx, sky)]
            + skeleton.calc_parallel_curves(
                skx, sky, distance=distance, ncurves=ncurves
            )[::-1]
        )

        plt.imshow(image[slice_number], cmap="gray")
        for curve in skeleton_curves:
            px, py = curve
            plt.plot(px, py)
        #plt.axes().set_aspect("equal", "datalim")
        plt.show()

        panoramic_skeleton_image = draw_bezier.planify_curves(image, np.array(curves))
        panoramic_skeleton_projection = panoramic_skeleton_image.max(0)
        vrint("panoramic_skeleton_image", [panoramic_skeleton_image.max(), \
                                           panoramic_skeleton_image.min(), \
                                           panoramic_skeleton_image.shape], verbose)
        plt.imshow(normalize_(panoramic_skeleton_projection, 255))
        plt.show()
        sx, sy, sz = spacing
        sx = (
            (
                (sx * skx[::2] - sx * skx[1::2]) ** 2
                + (sy * sky[::2] - sy * sky[1::2]) ** 2
            )
            ** 0.5
        ).mean()
        #imageio.imsave(f"{output_filename_panoramic}_skeleton_panoramic.png",
        #               (normalize_(panoramic_skeleton_image, 255).astype(np.uint8)))

        #save_image(
        #    panoramic_skeleton_image,
        #    str(output_filename_skeleton),
        #    spacing=(sx, sz, distance),
        #)


if __name__ == "__main__":
    main()
