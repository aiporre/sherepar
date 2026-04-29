# this script will use

import argparse
import os

import numpy as np

from spherepar.benchmark.spheres_generator import (
    generate_random_ellipsoid_points,
    create_watertight_mesh,
    save_to_obj,
    plot_ellipsoid,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a watertight sphere mesh."
    )
    parser.add_argument("--r", type=float, default=2, help="radius r.")
    parser.add_argument(
        "--n-meshes",
        type=int,
        default=10,
        dest="num_spheres",
        help="number of spheres to generate.",
    )
    parser.add_argument(
        "--num-point",
        "--num_points",
        dest="num_point",
        type=int,
        default=400,
        help="Number of sampled surface points.",
    )
    parser.add_argument(
        "--file-name-template",
        "--file_name_template",
        default="sphere_*.obj",
        dest="file_name_template",
        type=str,
        help=(
            "File name template for the spheres file, use '*' or '{}' as placeholder for index,"
            " e.g. 'sphere_*.obj' -> sphere_C000.obj, sphere_C001.obj ..."
        ),
    )
    # output path:
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        type=str,
        default="./spheres_templates",
        help =(
            "Output directory to save the generated sphere OBJ files. Default is './spheres_templates'."
            " The directory will be created if it does not exist."
        )
    )
    parser.add_argument(
        "--plot-mode",
        dest="plot_mode",
        choices=["save", "show"],
        default="save",
        help=("Plot mode: 'save' will write a PNG to the output directory (default). "
              "show' will open an interactive window.")
    )
    parser.add_argument(
        "--dpi",
        dest="dpi",
        type=int,
        default=200,
        help="DPI to use when saving PNG plots (default: 200)",
    )
    parser.add_argument(
        "--fig-size",
        dest="fig_size",
        type=str,
        default="12x9",
        help=("Figure size WxH in inches, format WIDTHxHEIGHT (default '12x9')."),
    )
    parser.add_argument(
        "--plots-subdir",
        dest="plots_subdir",
        type=str,
        default=None,
        help=("Subdirectory under the output dir to store plots. If omitted, PNGs are saved "
              "next to the OBJ files (i.e. same as output dir)."),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("Generating random points on ellipsoid surface...")
    r = args.r
    print(f"Generating spheres:")
    print(f"Radius: {r}")
    print(f"Number of spheres: {args.num_spheres}")
    print(f"Number of points per sphere: {args.num_point}")
    print(f"Filename template: {args.file_name_template}")

    # ensure output folder exists
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # determine where to save plots: if user supplied a plots_subdir, use it under out_dir,
    # otherwise PNGs are placed in the same directory as the OBJ files (out_dir)
    if args.plots_subdir:
        plots_dir = os.path.join(out_dir, args.plots_subdir)
    else:
        plots_dir = out_dir
    if args.plot_mode == "save":
        os.makedirs(plots_dir, exist_ok=True)

    def format_filename(template: str, idx: int) -> str:
        tag = f"C{idx:03d}"
        if "*" in template:
            return template.replace("*", tag)
        if "{}" in template:
            return template.format(tag)
        # no placeholder: insert before extension
        name, ext = os.path.splitext(template)
        return f"{name}_{tag}{ext}"

    for sphere_id in range(args.num_spheres):
        print(f"\nGenerating sphere {sphere_id}/{args.num_spheres - 1}...")

        # Generate random points on sphere for this instance
        vertices = generate_random_ellipsoid_points(
            a=r, b=r, c=r, num_points=args.num_point
        )

        print("Creating watertight mesh using convex hull...")
        vertices_watertight, faces = create_watertight_mesh(vertices)
        print(f"Created watertight mesh with {len(faces)} triangular faces")

        # Save to OBJ file
        out_name = format_filename(args.file_name_template, sphere_id)
        out_path = os.path.join(out_dir, out_name)
        print(f"Saving to OBJ file: {out_path}")
        save_to_obj(out_path, vertices_watertight, faces)

        # Plot or save the mesh figure
        print("Plotting watertight ellipsoid (may open a window or save PNG)...")
        try:
            if args.plot_mode == "show":
                # delegate to helper that shows interactive window
                plot_ellipsoid(vertices_watertight, faces)
            else:
                # create a PNG in the plots directory
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection

                # parse figure size passed as WIDTHxHEIGHT
                try:
                    w, h = [float(x) for x in args.fig_size.lower().split("x")]
                except Exception:
                    w, h = 12.0, 9.0

                fig = plt.figure(figsize=(w, h))
                ax = fig.add_subplot(111, projection='3d')
                triangle_vertices = vertices_watertight[faces]
                poly = Poly3DCollection(triangle_vertices, alpha=0.7, linewidths=0.3, edgecolors='black')
                poly.set_facecolor([0.2, 0.6, 0.9])
                poly.set_edgecolor('lightgray')
                ax.add_collection3d(poly)
                ax.scatter(vertices_watertight[:, 0], vertices_watertight[:, 1], vertices_watertight[:, 2],
                           c='red', s=6, alpha=0.6)

                max_range = max(
                    np.ptp(vertices_watertight[:, 0]),
                    np.ptp(vertices_watertight[:, 1]),
                    np.ptp(vertices_watertight[:, 2])
                ) / 2.0
                mid_x = (vertices_watertight[:, 0].max() + vertices_watertight[:, 0].min()) * 0.5
                mid_y = (vertices_watertight[:, 1].max() + vertices_watertight[:, 1].min()) * 0.5
                mid_z = (vertices_watertight[:, 2].max() + vertices_watertight[:, 2].min()) * 0.5
                ax.set_xlim(mid_x - max_range, mid_x + max_range)
                ax.set_ylim(mid_y - max_range, mid_y + max_range)
                ax.set_zlim(mid_z - max_range, mid_z + max_range)

                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')

                out_png = os.path.join(plots_dir, f"{os.path.splitext(out_name)[0]}.png")
                plt.tight_layout()
                fig.savefig(out_png, dpi=args.dpi, bbox_inches='tight')
                plt.close(fig)
                print(f"Saved plot to {out_png}")
        except Exception as e:
            # plotting is optional; continue if it fails in headless env
            print("ERROR: ", e)
            print("Plot failed. Continuing...")


if __name__ == "__main__":
    main()
