import argparse
import os

from spherepar.benchmark.spheres_generator import generate_random_ellipsoid_points, create_watertight_mesh, save_to_obj, \
    plot_ellipsoid


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a watertight ellipsoid mesh."
    )
    parser.add_argument("--a", type=float, default=2, help="Semi-axis length along x.")
    parser.add_argument("--b", type=float, default=1.5, help="Semi-axis length along y.")
    parser.add_argument("--c", type=float, default=1, help="Semi-axis length along z.")
    parser.add_argument(
        "--num-point",
        "--num_points",
        dest="num_point",
        type=int,
        default=400,
        help="Number of sampled surface points.",
    )
    parser.add_argument(
        "--file-name",
        "--file_name",
        default="ellipsoid.obj",
        dest="file_name",
        type=str,
        help="File name of the ellipsoid file in ../data"
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        type=str,
        default="../data",
        help="Output directory to save the OBJ and optional PNG (default: ../data)",
    )
    parser.add_argument(
        "--plot-mode",
        dest="plot_mode",
        choices=["save", "show"],
        default="save",
        help="Plot mode: 'save' writes PNG, 'show' opens interactive window (default: save)",
    )
    parser.add_argument(
        "--dpi",
        dest="dpi",
        type=int,
        default=200,
        help="DPI for saved PNGs (default 200)",
    )
    parser.add_argument(
        "--fig-size",
        dest="fig_size",
        type=str,
        default="12x9",
        help="Figure size WIDTHxHEIGHT in inches for saved PNGs (default 12x9)",
    )
    parser.add_argument(
        "--plots-subdir",
        dest="plots_subdir",
        type=str,
        default=None,
        help="Subdirectory under output dir to store plots; if omitted, PNGs are saved next to the OBJ file",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("Generating random points on ellipsoid surface...")

    # Generate random points on ellipsoid
    vertices = generate_random_ellipsoid_points(
        a=args.a,
        b=args.b,
        c=args.c,
        num_points=args.num_point,
    )
    print(f"Generated {len(vertices)} random points")

    print("\nCreating watertight mesh using convex hull...")
    vertices, faces = create_watertight_mesh(vertices)
    print(f"Created watertight mesh with {len(faces)} triangular faces")

    # Ensure output directory exists
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # Save to OBJ file
    out_path = os.path.join(out_dir, args.file_name)
    print("\nSaving to OBJ file...", out_path)
    save_to_obj(out_path, vertices, faces)

    # Prepare plots directory
    if args.plots_subdir:
        plots_dir = os.path.join(out_dir, args.plots_subdir)
    else:
        plots_dir = out_dir
    if args.plot_mode == "save":
        os.makedirs(plots_dir, exist_ok=True)

    # Plot the mesh (show or save)
    print("\nPlotting watertight ellipsoid...")
    try:
        if args.plot_mode == "show":
            plot_ellipsoid(vertices, faces)
        else:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection

            try:
                w, h = [float(x) for x in args.fig_size.lower().split("x")]
            except Exception:
                w, h = 12.0, 9.0

            fig = plt.figure(figsize=(w, h))
            ax = fig.add_subplot(111, projection='3d')
            triangle_vertices = vertices[faces]
            poly = Poly3DCollection(triangle_vertices, alpha=0.7, linewidths=0.3, edgecolors='black')
            poly.set_facecolor([0.2, 0.6, 0.9])
            poly.set_edgecolor('lightgray')
            ax.add_collection3d(poly)
            ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c='red', s=20, alpha=0.5)

            max_range = max(
                vertices[:, 0].ptp(), vertices[:, 1].ptp(), vertices[:, 2].ptp()
            ) / 2.0
            mid_x = (vertices[:, 0].max() + vertices[:, 0].min()) * 0.5
            mid_y = (vertices[:, 1].max() + vertices[:, 1].min()) * 0.5
            mid_z = (vertices[:, 2].max() + vertices[:, 2].min()) * 0.5
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)

            out_png = os.path.join(plots_dir, f"{os.path.splitext(args.file_name)[0]}.png")
            plt.tight_layout()
            fig.savefig(out_png, dpi=args.dpi, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved plot to {out_png}")
    except Exception:
        print("Plot failed (probably headless). Continuing...")


if __name__ == "__main__":
    main()
